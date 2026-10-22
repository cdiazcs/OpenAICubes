import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;


public class CentralServer {
    private static final int MAX_BYTES = 2_000_000;
    private static final int WORKER_TIMEOUT_MS = timeout("worker", 10000);
    private static final int IDENTIFY_TIMEOUT_MS = timeout("identify", 12000);
    private static final int CLIENT_TIMEOUT_MS = timeout("client", 30000);
    private static final Pattern DECIMAL = Pattern.compile("[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?");
    private static final AtomicInteger cursor = new AtomicInteger();
    private static final List<Node> workers = new ArrayList<>();
    private static DistributedCnn distributed;
    private static final ScheduledThreadPoolExecutor deadlines = new ScheduledThreadPoolExecutor(1, task -> {
        Thread thread = new Thread(task, "socket-deadlines");
        thread.setDaemon(true);
        return thread;
    });
    static { deadlines.setRemoveOnCancelPolicy(true); }

    private static class Node {
        final InetSocketAddress endpoint;
        volatile long skipIdentifyUntil;
        Node(String host, int port) { endpoint = new InetSocketAddress(host, port); }
        public String toString() { return endpoint.toString(); }
    }

    private static class InvalidWorkerResponse extends IOException {
        InvalidWorkerResponse() { super("invalid_worker_response"); }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Uso: java CentralServer <puerto> <hostWorker:puerto> [otros...]");
            return;
        }
        int port = Integer.parseInt(args[0]);
        for (int i = 1; i < args.length; i++) {
            int idx = args[i].lastIndexOf(':');
            if (idx < 1) throw new IllegalArgumentException("worker debe ser host:puerto");
            workers.add(new Node(args[i].substring(0,idx), Integer.parseInt(args[i].substring(idx+1))));
        }
        List<InetSocketAddress> endpoints = new ArrayList<>();
        for (Node node : workers) endpoints.add(node.endpoint);
        distributed = new DistributedCnn(endpoints, WORKER_TIMEOUT_MS, IDENTIFY_TIMEOUT_MS);
        int maxClients = Integer.getInteger("openaicubes.maxClients", 64);
        if (maxClients < 1 || maxClients > 1024) throw new IllegalArgumentException("invalid_max_clients");
        // Sin cola ilimitada: una conexion persistente no retiene solicitudes nuevas en espera indefinida.
        ExecutorService clients = new ThreadPoolExecutor(Math.min(16, maxClients), maxClients,
            30, TimeUnit.SECONDS, new SynchronousQueue<Runnable>());
        try (ServerSocket listener = new ServerSocket(port)) {
            System.out.println("CentralServer en " + port + ", workers=" + workers);
            while (true) {
                Socket socket = listener.accept();
                try { clients.execute(() -> handle(socket)); }
                catch (RejectedExecutionException busy) {
                    socket.close();
                    System.err.println("Conexion rechazada: client_limit");
                }
            }
        } finally { clients.shutdown(); distributed.close(); deadlines.shutdownNow(); }
    }

    private static void handle(Socket socket) {
        try (Socket client = socket;
             DataInputStream input = new DataInputStream(new BufferedInputStream(client.getInputStream()));
             DataOutputStream output = new DataOutputStream(new BufferedOutputStream(client.getOutputStream()))) {
            client.setSoTimeout(CLIENT_TIMEOUT_MS);
            while (true) {
                // El cierre programado tambien interrumpe escrituras bloqueadas o un peer lento.
                ScheduledFuture<?> guard = closeAfter(client, CLIENT_TIMEOUT_MS);
                try {
                    String request;
                    try { request = read(input); }
                    catch (EOFException ex) { return; }
                    String[] fields = request.split("\\|", -1);
                    String id = fields.length > 1 ? fields[1] : "unknown";
                    String reply;
                    if (fields[0].equals("CONFIG") && fields.length == 2 && validId(id)) {
                        try { reply = distributed.configuration(id); }
                        catch (DistributedCnn.Failure failure) { reply = "ERROR|"+id+"|"+failure.code; }
                    } else if (fields[0].equals("IDENTIFY")) {
                        String invalid = validateIdentify(fields);
                        if (invalid != null) reply = invalid;
                        else if (fields.length == 4 && fields[3].equals("DISTRIBUTED")) {
                            try { reply = distributed.identify(id, fields[2]); }
                            catch (DistributedCnn.Failure failure) { reply = "ERROR|" + id + "|" + failure.code; }
                        } else reply = identify("IDENTIFY|" + id + "|" + fields[2], id);
                    } else if (fields.length >= 2 && Arrays.asList("PING", "CONV", "MLP").contains(fields[0])) {
                        try { reply = forward(request); }
                        catch (IOException ex) { reply = "ERROR|" + id + "|no_worker_available"; }
                    } else reply = "ERROR|" + id + "|unsupported_command";
                    write(output, reply);
                } finally { guard.cancel(false); }
            }
        } catch (IOException ex) { System.err.println("Cliente desconectado: " + ex.getMessage()); }
    }

    private static String forward(String request) throws IOException {
        int first = Math.floorMod(cursor.getAndIncrement(), workers.size());
        IOException last = new IOException("no_worker_available");
        for (int k = 0; k < workers.size(); k++) {
            InetSocketAddress endpoint = workers.get((first + k) % workers.size()).endpoint;
            try (Socket worker = new Socket()) {
                ScheduledFuture<?> guard = closeAfter(worker, WORKER_TIMEOUT_MS);
                try {
                    worker.connect(endpoint, Math.min(2000, WORKER_TIMEOUT_MS));
                    worker.setSoTimeout(WORKER_TIMEOUT_MS);
                    DataOutputStream out = new DataOutputStream(new BufferedOutputStream(worker.getOutputStream()));
                    DataInputStream in = new DataInputStream(new BufferedInputStream(worker.getInputStream()));
                    write(out, request);
                    return read(in);
                } finally { guard.cancel(false); }
            } catch (IOException ex) {
                last = ex;
                System.err.println("Worker falló: " + endpoint + " : " + ex.getMessage());
            }
        }
        throw last;
    }

    private static String identify(String request, String id) {
        long end = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(IDENTIFY_TIMEOUT_MS);
        String failure = "no_identify_worker_available";
        for (Node node : workers) {
            if (node.skipIdentifyUntil != 0 && node.skipIdentifyUntil - System.nanoTime() > 0) continue;
            long remaining = TimeUnit.NANOSECONDS.toMillis(end - System.nanoTime());
            if (remaining <= 0) return "ERROR|" + id + "|identify_worker_timeout";
            int attemptMs = (int)Math.min(remaining, WORKER_TIMEOUT_MS);
            long attemptEnd = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(attemptMs);
            try (Socket worker = new Socket()) {
                ScheduledFuture<?> guard = closeAfter(worker, attemptMs);
                try {
                    worker.connect(node.endpoint, Math.min(2000, attemptMs));
                    worker.setSoTimeout(Math.min(2000, attemptMs));
                    DataOutputStream out = new DataOutputStream(new BufferedOutputStream(worker.getOutputStream()));
                    DataInputStream in = new DataInputStream(new BufferedInputStream(worker.getInputStream()));
                    // PING en la misma conexion evita usar una capacidad antigua tras un reinicio.
                    write(out, "PING|" + id);
                    String[] pong = read(in).split("\\|", -1);
                    if (pong.length != 3 || !pong[0].equals("PONG") || !pong[1].equals(id))
                        throw new InvalidWorkerResponse();
                    if (!pong[2].equals("Python")) {
                        node.skipIdentifyUntil = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                        continue;
                    }
                    worker.setSoTimeout(attemptMs);
                    write(out, request);
                    String reply = read(in);
                    validateIdentifyReply(reply, id);
                    if (reply.equals("ERROR|" + id + "|unsupported_command")) {
                        node.skipIdentifyUntil = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                        continue;
                    }
                    return reply;
                } finally { guard.cancel(false); }
            } catch (InvalidWorkerResponse ex) {
                failure = "invalid_worker_response";
            } catch (SocketTimeoutException ex) {
                failure = "identify_worker_timeout";
            } catch (IOException ex) {
                if (System.nanoTime() - attemptEnd >= 0) failure = "identify_worker_timeout";
                System.err.println("IDENTIFY worker fallo: " + node.endpoint + " : " + ex.getMessage());
            }
        }
        return "ERROR|" + id + "|" + failure;
    }

    private static String validateIdentify(String[] fields) {
        if (fields.length < 2 || !validId(fields[1])) return "ERROR|unknown|invalid_request_id";
        String prefix = "ERROR|" + fields[1] + "|";
        if (fields.length != 3 && fields.length != 4) return prefix + "invalid_identify_fields";
        if (fields.length == 4 && !fields[3].equals("SEQUENTIAL") && !fields[3].equals("DISTRIBUTED"))
            return prefix + "invalid_identify_fields";
        String[] pixels = fields[2].split(",", -1);
        if (pixels.length != 256) return prefix + "invalid_image_size";
        for (String pixel : pixels) if (!normalizedNumber(pixel)) return prefix + "invalid_pixels";
        return null;
    }

    private static boolean validId(String id) {
        if (id.isEmpty() || id.codePointCount(0, id.length()) > 128) return false;
        for (int at = 0; at < id.length();) {
            int cp = id.codePointAt(at);
            if (cp == '|' || Character.isISOControl(cp) || Character.isWhitespace(cp)
                    || Character.isSpaceChar(cp) || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
            at += Character.charCount(cp);
        }
        return true;
    }

    private static boolean normalizedNumber(String raw) {
        if (raw.length() > 64 || !DECIMAL.matcher(raw).matches()) return false;
        double value = Double.parseDouble(raw);
        return Double.isFinite(value) && value >= 0 && value <= 1;
    }

    private static void validateIdentifyReply(String reply, String id) throws IOException {
        String[] fields = reply.split("\\|", -1);
        if (fields.length == 3 && fields[0].equals("ERROR") && fields[1].equals(id)
                && fields[2].matches("[A-Za-z0-9_:-]{1,128}")) return;
        if (fields.length != 5 || !fields[0].equals("RESULT") || !fields[1].equals(id)
                || !fields[2].equals("1") || !fields[3].equals("3")) throw new InvalidWorkerResponse();
        String[] values = fields[4].split(",", -1);
        if (values.length != 3) throw new InvalidWorkerResponse();
        double sum = 0;
        for (String value : values) {
            if (!normalizedNumber(value)) throw new InvalidWorkerResponse();
            sum += Double.parseDouble(value);
        }
        if (Math.abs(sum - 1) > 1e-8) throw new InvalidWorkerResponse();
    }

    private static ScheduledFuture<?> closeAfter(Socket socket, long delayMs) {
        return deadlines.schedule(() -> {
            try { socket.close(); } catch (IOException ignored) { }
        }, Math.max(1, delayMs), TimeUnit.MILLISECONDS);
    }

    private static int timeout(String name, int fallback) {
        int value = Integer.getInteger("openaicubes." + name + "TimeoutMs", fallback);
        if (value < 1 || value > 120000) throw new IllegalArgumentException("invalid_timeout");
        return value;
    }

    private static String read(DataInputStream in) throws IOException {
        int len = in.readInt();
        if (len <= 0 || len > MAX_BYTES) throw new IOException("invalid_length");
        byte[] data = new byte[len];
        in.readFully(data);
        return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(data)).toString();
    }
    private static void write(DataOutputStream out, String s) throws IOException {
        byte[] data = s.getBytes(StandardCharsets.UTF_8);
        if (data.length > MAX_BYTES) throw new IOException("too_large");
        out.writeInt(data.length);
        out.write(data);
        out.flush();
    }
}
