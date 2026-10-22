import java.io.*;
import java.net.*;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.Consumer;


public final class ImageClient implements AutoCloseable {
    public enum Mode { SEQUENTIAL, DISTRIBUTED }
    public enum Phase { CONNECTING, SENDING, PROCESSING, RECEIVED }
    public static final class Identity {
        private final String session = UUID.randomUUID().toString();
        private final AtomicLong sequence = new AtomicLong();
        public final String clientId = "demo-"+session;
        public String nextRequestId() { return "req-"+session+"-"+sequence.incrementAndGet(); }
    }
    public static final class Response {
        public final String requestId;
        public final Mode mode;
        public final ClientCli.Prediction prediction;
        public final double millis;
        private Response(String id, Mode mode, ClientCli.Prediction prediction, double millis) {
            this.requestId = id; this.mode = mode; this.prediction = prediction; this.millis = millis;
        }
    }
    private static final ScheduledThreadPoolExecutor DEADLINES = new ScheduledThreadPoolExecutor(1, task -> {
        Thread thread = new Thread(task, "image-client-deadlines"); thread.setDaemon(true); return thread;
    });
   
    private static final ExecutorService RESOLVER = new ThreadPoolExecutor(2, 2, 0, TimeUnit.SECONDS,
        new ArrayBlockingQueue<Runnable>(16), task -> {
            Thread thread = new Thread(task, "image-client-dns"); thread.setDaemon(true); return thread;
        });
    static { DEADLINES.setRemoveOnCancelPolicy(true); }
    private final String host;
    private final int port, timeoutMs;
    private final AtomicBoolean closed = new AtomicBoolean(), used = new AtomicBoolean();
    private volatile Socket socket;

    public ImageClient(String host, int port, int timeoutMs) {
        if (host == null || host.trim().isEmpty() || host.length() > 253
                || host.chars().anyMatch(c -> Character.isWhitespace(c) || Character.isISOControl(c)))
            throw new IllegalArgumentException("invalid_host");
        if (port < 1 || port > 65535) throw new IllegalArgumentException("invalid_port");
        if (timeoutMs < 1 || timeoutMs > 120000) throw new IllegalArgumentException("invalid_timeout");
        this.host = host; this.port = port; this.timeoutMs = timeoutMs;
    }

    public static String message(String requestId, double[] pixels, Mode mode) {
        if (!ClientCli.validId(requestId)) throw new IllegalArgumentException("invalid_request_id");
        if (mode == null) throw new IllegalArgumentException("invalid_mode");
        if (pixels == null || pixels.length != 256) throw new IllegalArgumentException("invalid_image_size");
        StringBuilder text = new StringBuilder("IDENTIFY|"+requestId+"|");
        for (int i = 0; i < pixels.length; i++) {
            double value = pixels[i];
            if (!Double.isFinite(value) || value < 0 || value > 1) throw new IllegalArgumentException("invalid_pixels");
            if (i > 0) text.append(',');
            text.append(Double.toString(value));
        }
        return text.append('|').append(mode.name()).toString();
    }

    public Response identify(String id, double[] pixels, Mode mode, Consumer<Phase> progress) throws IOException {
        if (!used.compareAndSet(false, true)) throw new IllegalStateException("client_already_used");
        if (closed.get()) throw new InterruptedIOException("request_cancelled");
        String request = message(id, pixels, mode);
        long started = System.nanoTime(), end = started+TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        AtomicBoolean timedOut = new AtomicBoolean();
        Socket connection = new Socket();
        socket = connection;
        if (closed.get()) { close(); throw new InterruptedIOException("request_cancelled"); }
        ScheduledFuture<?> guard = DEADLINES.schedule(() -> { timedOut.set(true); close(); }, timeoutMs, TimeUnit.MILLISECONDS);
        try (Socket owned = connection) {
            progress.accept(Phase.CONNECTING);
            InetAddress address = resolve(end);
            owned.connect(new InetSocketAddress(address, port), Math.min(5000, remaining(end)));
            owned.setSoTimeout(remaining(end));
            owned.setTcpNoDelay(true);
            DataOutputStream out = new DataOutputStream(new BufferedOutputStream(owned.getOutputStream()));
            DataInputStream in = new DataInputStream(new BufferedInputStream(owned.getInputStream()));
            progress.accept(Phase.SENDING);
            String reply = ClientCli.exchange(in, out, request, () -> progress.accept(Phase.PROCESSING));
            ClientCli.Prediction prediction = ClientCli.parsePrediction(reply, id);
            if (timedOut.get() || System.nanoTime()-end >= 0) throw new SocketTimeoutException("response_timeout");
            progress.accept(Phase.RECEIVED);
            return new Response(id, mode, prediction, (System.nanoTime()-started)/1_000_000.0);
        } catch (IOException error) {
            if (timedOut.get()) throw new SocketTimeoutException("response_timeout");
            if (closed.get()) throw new InterruptedIOException("request_cancelled");
            throw error;
        } finally { guard.cancel(false); close(); }
    }

    private InetAddress resolve(long end) throws IOException {
        Future<InetAddress> future;
        try { future = RESOLVER.submit(() -> InetAddress.getByName(host)); }
        catch (RejectedExecutionException error) { throw new IOException("resolver_busy", error); }
        try { return future.get(remaining(end), TimeUnit.MILLISECONDS); }
        catch (TimeoutException error) { throw new SocketTimeoutException("response_timeout"); }
        catch (InterruptedException error) {
            Thread.currentThread().interrupt(); throw new InterruptedIOException("request_cancelled");
        } catch (ExecutionException error) {
            if (error.getCause() instanceof IOException) throw (IOException)error.getCause();
            throw new IOException("invalid_host", error.getCause());
        } finally { future.cancel(true); }
    }
    private static int remaining(long end) throws SocketTimeoutException {
        long delta = end-System.nanoTime();
        if (delta <= 0) throw new SocketTimeoutException("response_timeout");
        return (int)Math.max(1, TimeUnit.NANOSECONDS.toMillis(delta));
    }
    @Override public void close() {
        closed.set(true);
        Socket current = socket;
        if (current != null) try { current.close(); } catch (IOException ignored) { }
    }

    public static String explain(Throwable error) {
        if (error instanceof ClientCli.ServerError) {
            String code = ((ClientCli.ServerError)error).code;
            switch (code) {
                case "insufficient_workers": return "El modo distribuido necesita al menos dos workers disponibles.";
                case "no_identify_worker_available": return "No hay un worker Python disponible para el modo secuencial.";
                case "model_unavailable": return "El servidor no pudo cargar los pesos del modelo.";
                case "worker_timeout": case "distributed_timeout": return "Los workers no terminaron dentro del tiempo límite.";
                case "coordinator_busy": case "worker_busy": return "El servidor está ocupado. Intenta de nuevo en unos segundos.";
                default: return "El servidor rechazó la consulta ("+code+").";
            }
        }
        if (error instanceof SocketTimeoutException) return "Se agotó el tiempo de espera. Revisa el servidor o aumenta el límite.";
        if (error instanceof ConnectException) return "No se pudo conectar. Revisa el host, el puerto y que el servidor esté iniciado.";
        if (error instanceof UnknownHostException) return "No se encontró el host. Revisa la dirección del servidor.";
        if (error instanceof EOFException) return "El servidor cerró la conexión antes de completar la respuesta.";
        if (error instanceof InterruptedIOException) return "Consulta cancelada.";
        String code = error.getMessage() == null ? "" : error.getMessage();
        if (code.startsWith("invalid_image") || code.contains("pgm") || code.equals("unsupported_image_format"))
            return "No se pudo leer la imagen. Selecciona un PNG, JPEG o PGM válido (hasta 32 MB y 4 megapíxeles).";
        if (code.equals("invalid_host") || code.equals("invalid_port") || code.equals("invalid_timeout"))
            return "Revisa el host, el puerto (1–65535) y el tiempo límite (1–120 segundos).";
        if (code.startsWith("invalid_") || error instanceof java.nio.charset.CharacterCodingException)
            return "La respuesta del servidor no es válida para esta consulta.";
        return "No se pudo completar la consulta. Revisa la conexión y vuelve a intentarlo.";
    }
}
