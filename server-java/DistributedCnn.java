import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Pattern;


public final class DistributedCnn implements AutoCloseable {
    private static final int MAX_BYTES = 2_000_000;
    private static final Pattern DECIMAL = Pattern.compile("[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?");
    private final List<InetSocketAddress> endpoints;
    private final ExecutorService tasks;
    private final ScheduledThreadPoolExecutor deadlines;
    private final Set<Run> active = ConcurrentHashMap.newKeySet();
    // Inicializar la entropia antes de aceptar conexiones: en Windows puede ser
    // lento el primer UUID. El contador evita colisiones dentro de este servidor.
    private final String runPrefix = UUID.randomUUID().toString();
    private final AtomicLong runSequence = new AtomicLong();
    private final int workerTimeoutMs, requestTimeoutMs, denseChunkSize;
    private final boolean traceValues;
    private final boolean remotePostprocess = Boolean.getBoolean("openaicubes.remotePostprocess");
    private volatile Model model;
    private final AtomicBoolean closed = new AtomicBoolean();

    public static final class Failure extends Exception {
        public final String code;
        Failure(String code) { super(code); this.code = code; }
    }

    private static final class Node {
        final InetSocketAddress endpoint;
        final String language;
        final Set<String> operations;
        Node(InetSocketAddress endpoint, String language, String operations) {
            this.endpoint = endpoint; this.language = language;
            this.operations = new HashSet<>(Arrays.asList(operations.split(",", -1)));
        }
    }

    private static final class Run {
        final String clientId, id;
        final long end;
        final AtomicBoolean cancelled = new AtomicBoolean();
        final Set<Socket> sockets = ConcurrentHashMap.newKeySet();
        final Set<InetSocketAddress> failed = ConcurrentHashMap.newKeySet();
        final Set<InetSocketAddress> participants = ConcurrentHashMap.newKeySet();
        Run(String clientId, int timeout, String id) {
            this.clientId = clientId; this.id = id;
            end = System.nanoTime()+TimeUnit.MILLISECONDS.toNanos(timeout);
        }
        void register(Socket socket) throws Failure {
            sockets.add(socket);
            if (cancelled.get()) { cancel(); throw new Failure("distributed_cancelled"); }
        }
        void cancel() {
            cancelled.set(true);
            for (Socket socket : sockets) try { socket.close(); } catch (IOException ignored) { }
        }
    }

    private static final class Chunk {
        final String taskId, op, payload;
        final int offset, rows, cols, preferred;
        Chunk(String taskId, String op, String payload, int offset, int rows, int cols, int preferred) {
            this.taskId = taskId; this.op = op; this.payload = payload;
            this.offset = offset; this.rows = rows; this.cols = cols; this.preferred = preferred;
        }
    }
    private static final class Part {
        final Chunk chunk;
        final double[] values;
        Part(Chunk chunk, double[] values) { this.chunk = chunk; this.values = values; }
    }

    public DistributedCnn(List<InetSocketAddress> endpoints, int workerTimeoutMs, int requestTimeoutMs) {
        this.endpoints = Collections.unmodifiableList(new ArrayList<>(new LinkedHashSet<>(endpoints)));
        this.workerTimeoutMs = workerTimeoutMs;
        this.requestTimeoutMs = requestTimeoutMs;
        traceValues = Boolean.getBoolean("openaicubes.traceValues");
        int threads = property("taskThreads", 6, 1, 256);
        int queue = property("maxQueuedChunks", 512, 1, 65536);
        denseChunkSize = property("denseChunkSize", 0, 0, 8);
        tasks = new ThreadPoolExecutor(threads, threads, 0, TimeUnit.MILLISECONDS,
                                      new ArrayBlockingQueue<Runnable>(queue));
        deadlines = new ScheduledThreadPoolExecutor(1, runnable -> {
            Thread thread = new Thread(runnable, "distributed-deadlines");
            thread.setDaemon(true);
            return thread;
        });
        deadlines.setRemoveOnCancelPolicy(true);
    }

    public String identify(String clientId, String pixels) throws Failure {
        if (closed.get()) throw new Failure("coordinator_closed");
        Run run = new Run(clientId, requestTimeoutMs, runPrefix+"-"+runSequence.incrementAndGet());
        active.add(run);
        try {
            double[] image = numbers(pixels, 256);
            for (double x : image) if (x < 0 || x > 1) throw new Failure("invalid_pixels");
            Model weights = weights();
            List<Node> nodes = discover(run);
            List<Node> convNodes = capable(nodes, "CONV"), denseNodes = capable(nodes, "DENSE");
            List<Node> postNodes = capable(nodes, "POST");
            if (remotePostprocess && postNodes.size() < 2) throw new Failure("insufficient_workers");
            if (convNodes.size() < 2 || denseNodes.size() < 2) throw new Failure("insufficient_workers");

            List<Chunk> chunks = new ArrayList<>();
            for (int filter = 0; filter < 2; filter++) {
                String payload = "CONV|16|16|3|3|"+csv(image)+"|"+csv(weights.filters[filter]);
                chunks.add(new Chunk("conv-"+filter, "CONV", payload, filter, 14, 14, filter));
            }
            double[][] maps = new double[2][];
            for (Part part : layer(run, chunks, convNodes)) maps[part.chunk.offset] = part.values;
            double[] flat = new double[98];
            for (int channel = 0; channel < 2; channel++) {
                tensor(run, "conv-"+channel, 14, 14, maps[channel]);
                double[] pool = new double[49];
                if (remotePostprocess) pool = post(run, "pool-"+channel, "POOL", maps[channel], 14, 14, 7, 7, postNodes);
                else for (int r = 0; r < 7; r++) for (int c = 0; c < 7; c++) {
                    int p = (2*r)*14+2*c;
                    pool[r*7+c] = Math.max(0, Math.max(Math.max(maps[channel][p], maps[channel][p+1]),
                                                    Math.max(maps[channel][p+14], maps[channel][p+15])));
                }
                tensor(run, "pool-"+channel, 7, 7, pool);
                System.arraycopy(pool, 0, flat, channel*49, 49);
            }
            tensor(run, "flatten", 1, 98, flat);
            double[] hidden = dense(run, "hidden", flat, weights.w1, weights.b1, denseNodes, 2);
            tensor(run, "hidden-linear", 1, 8, hidden);
            if (remotePostprocess) hidden = post(run, "relu", "RELU", hidden, 1, 8, 1, 8, postNodes);
            else for (int i = 0; i < hidden.length; i++) hidden[i] = Math.max(0, hidden[i]);
            tensor(run, "hidden-relu", 1, 8, hidden);
            int hiddenTasks = (8+chunkSize(8, denseNodes.size())-1)/chunkSize(8, denseNodes.size());
            double[] logits = dense(run, "output", hidden, weights.w2, weights.b2, denseNodes, 2+hiddenTasks);
            tensor(run, "logits", 1, 3, logits);
            if (remotePostprocess) logits = post(run, "softmax", "SOFTMAX", logits, 1, 3, 1, 3, postNodes);
            else {
            double maximum = Math.max(logits[0], Math.max(logits[1], logits[2]));
            double total = 0;
            for (int i = 0; i < 3; i++) { logits[i] = Math.exp(logits[i]-maximum); total += logits[i]; }
            for (int i = 0; i < 3; i++) logits[i] /= total;
            }
            tensor(run, "probabilities", 1, 3, logits);
            if (run.participants.size() < 2) throw new Failure("insufficient_workers");
            return "RESULT|"+clientId+"|1|3|"+csv(logits);
        } finally {
            run.cancel();
            active.remove(run);
        }
    }

    private double[] post(Run run, String id, String kind, double[] input, int ir, int ic, int or, int oc, List<Node> nodes) throws Failure {
        Chunk chunk = new Chunk(id, "POST", "POST|"+kind+"|"+ir+"|"+ic+"|"+csv(input), 0, or, oc, 0);
        return layer(run, Collections.singletonList(chunk), nodes).get(0).values;
    }

    public String configuration(String id) throws Failure {
        StringBuilder registered = new StringBuilder();
        for (InetSocketAddress endpoint : endpoints) {
            if (registered.length() > 0) registered.append(',');
            registered.append(endpoint.getHostString()).append(':').append(endpoint.getPort());
        }
        try {
            byte[] bytes = Files.readAllBytes(weightsPath());
            StringBuilder hash = new StringBuilder();
            for (byte value : java.security.MessageDigest.getInstance("SHA-256").digest(bytes)) hash.append(String.format(Locale.ROOT, "%02x", value & 255));
            return "CONFIG|"+id+"|"+System.getProperty("openaicubes.benchmarkProfile", "NONE")+"|"+remotePostprocess
                +"|"+property("taskThreads", 6, 1, 256)+"|"+registered+"|"+hash+"|"+denseChunkSize;
        } catch (Exception error) { throw new Failure("model_unavailable"); }
    }

    private double[] dense(Run run, String name, double[] input, double[] weights,
                           double[] bias, List<Node> nodes, int firstNode) throws Failure {
        List<Chunk> chunks = new ArrayList<>();
        int block = chunkSize(bias.length, nodes.size());
        for (int from = 0; from < bias.length; from += block) {
            int to = Math.min(bias.length, from+block), count = to-from;
            String payload = "DENSE|"+input.length+"|"+count+"|"+csv(input)+"|"
                +csv(Arrays.copyOfRange(weights, from*input.length, to*input.length))+"|"
                +csv(Arrays.copyOfRange(bias, from, to));
            chunks.add(new Chunk(name+"-"+from+"-"+(to-1), "DENSE", payload, from, 1, count, firstNode+chunks.size()));
        }
        double[] result = new double[bias.length];
        for (Part part : layer(run, chunks, nodes))
            System.arraycopy(part.values, 0, result, part.chunk.offset, part.values.length);
        return result;
    }

    private int chunkSize(int count, int nodes) {
        int requested = denseChunkSize > 0 ? denseChunkSize : (count+nodes-1)/nodes;
        // Incluso con una configuracion grande se conservan dos chunks por capa.
        return Math.min(requested, (count+1)/2);
    }

    private List<Part> layer(Run run, List<Chunk> chunks, List<Node> nodes) throws Failure {
        CompletionService<Part> completion = new ExecutorCompletionService<>(tasks);
        List<Future<Part>> pending = new ArrayList<>();
        try {
            for (Chunk chunk : chunks) pending.add(completion.submit(() -> execute(run, chunk, nodes)));
            List<Part> parts = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                Future<Part> done = completion.poll(remaining(run), TimeUnit.MILLISECONDS);
                if (done == null) throw new Failure("distributed_timeout");
                parts.add(done.get()); // llegada arbitraria; el ensamblado usa offset, no llegada.
            }
            return parts;
        } catch (RejectedExecutionException ex) {
            throw new Failure("coordinator_busy");
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new Failure("distributed_cancelled");
        } catch (ExecutionException ex) {
            if (ex.getCause() instanceof Failure) throw (Failure)ex.getCause();
            throw new Failure("distributed_processing_error");
        } finally {
            for (Future<Part> future : pending) if (!future.isDone()) future.cancel(true);
        }
    }

    private Part execute(Run run, Chunk chunk, List<Node> nodes) throws Failure {
        Failure last = new Failure("no_chunk_worker_available");
        for (int attempt = 0; attempt < nodes.size(); attempt++) {
            Node node = nodes.get((chunk.preferred+attempt) % nodes.size());
            if (run.failed.contains(node.endpoint)) continue;
            long started = System.nanoTime();
            try {
                String message = "JOB|"+run.id+"|"+chunk.taskId+"|"+chunk.payload;
                String[] fields = exchange(run, node.endpoint, message, workerTimeoutMs).split("\\|", -1);
                if (fields.length == 4 && fields[0].equals("ERROR") && fields[1].equals(run.id)
                        && fields[2].equals(chunk.taskId) && fields[3].matches("[a-z_]{1,64}"))
                    throw new Failure(fields[3]);
                if (fields.length != 7 || !fields[0].equals("PART") || !fields[1].equals(run.id)
                        || !fields[2].equals(chunk.taskId) || !fields[3].equals(Integer.toString(chunk.rows))
                        || !fields[4].equals(Integer.toString(chunk.cols))) throw new Failure("invalid_partial_result");
                double[] values = numbers(fields[5], chunk.rows*chunk.cols);
                long compute;
                try { compute = Long.parseLong(fields[6]); }
                catch (NumberFormatException ex) { throw new Failure("invalid_partial_result"); }
                if (compute < 0 || compute > TimeUnit.MINUTES.toNanos(10)) throw new Failure("invalid_partial_result");
                run.participants.add(node.endpoint);
                taskLog(run, chunk, node, compute, started, "OK");
                return new Part(chunk, values);
            } catch (Failure failure) {
                last = failure;
                taskLog(run, chunk, node, 0, started, "FAILED:"+failure.code);
                if (run.cancelled.get() || System.nanoTime()-run.end >= 0) throw failure;
                run.failed.add(node.endpoint);
            }
        }
        throw last;
    }

    private List<Node> discover(Run run) throws Failure {
        List<Node> found = new ArrayList<>();
        for (InetSocketAddress endpoint : endpoints) {
            try {
                String[] caps = exchange(run, endpoint, "CAPS|"+run.id, Math.min(1000, workerTimeoutMs)).split("\\|", -1);
                if (caps.length != 5 || !caps[0].equals("CAPS") || !caps[1].equals(run.id)
                        || !Arrays.asList("Java", "Python", "CSharp").contains(caps[2])) continue;
                int threads;
                try { threads = Integer.parseInt(caps[4]); } catch (NumberFormatException ex) { continue; }
                if (threads < 1 || threads > 256) continue;
                found.add(new Node(endpoint, caps[2], caps[3]));
                System.out.println("NODE|"+run.clientId+"|"+run.id+"|"+endpoint+"|"+caps[2]+"|"+caps[3]+"|"+threads);
            } catch (Failure failure) {
                if (run.cancelled.get() || System.nanoTime()-run.end >= 0) throw failure;
            }
        }
        return found;
    }

    private static List<Node> capable(List<Node> nodes, String operation) {
        List<Node> result = new ArrayList<>();
        for (Node node : nodes) if (node.operations.contains(operation)) result.add(node);
        return result;
    }

    private String exchange(Run run, InetSocketAddress endpoint, String message, int limit) throws Failure {
        int timeout = Math.min(limit, remaining(run));
        long attemptEnd = System.nanoTime()+TimeUnit.MILLISECONDS.toNanos(timeout);
        try (Socket socket = new Socket()) {
            run.register(socket);
            ScheduledFuture<?> guard = deadlines.schedule(() -> {
                try { socket.close(); } catch (IOException ignored) { }
            }, timeout, TimeUnit.MILLISECONDS);
            try {
                socket.connect(endpoint, Math.min(2000, timeout));
                socket.setSoTimeout(timeout);
                socket.setTcpNoDelay(true);
                DataOutputStream out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
                byte[] body = message.getBytes(StandardCharsets.UTF_8);
                if (body.length == 0 || body.length > MAX_BYTES) throw new Failure("invalid_job_size");
                out.writeInt(body.length); out.write(body); out.flush();
                DataInputStream in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
                int length = in.readInt();
                if (length <= 0 || length > MAX_BYTES) throw new Failure("invalid_partial_result");
                byte[] data = new byte[length];
                in.readFully(data);
                return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(data)).toString();
            } finally { guard.cancel(false); run.sockets.remove(socket); }
        } catch (IOException ex) {
            if (run.cancelled.get()) throw new Failure("distributed_cancelled");
            if (System.nanoTime()-run.end >= 0) throw new Failure("distributed_timeout");
            if (ex instanceof SocketTimeoutException || System.nanoTime()-attemptEnd >= 0) throw new Failure("worker_timeout");
            throw new Failure("node_unavailable");
        }
    }

    private static int remaining(Run run) throws Failure {
        if (run.cancelled.get() || Thread.currentThread().isInterrupted()) throw new Failure("distributed_cancelled");
        long delta = run.end-System.nanoTime();
        if (delta <= 0) throw new Failure("distributed_timeout");
        return (int)Math.max(1, TimeUnit.NANOSECONDS.toMillis(delta));
    }

    private static double[] numbers(String raw, int count) throws Failure {
        String[] tokens = raw.split(",", -1);
        if (tokens.length != count) throw new Failure("invalid_partial_result");
        double[] result = new double[count];
        for (int i = 0; i < count; i++) {
            if (tokens[i].length() > 64 || !DECIMAL.matcher(tokens[i]).matches()) throw new Failure("invalid_partial_result");
            result[i] = Double.parseDouble(tokens[i]);
            if (!Double.isFinite(result[i])) throw new Failure("invalid_partial_result");
        }
        return result;
    }
    private static String csv(double[] values) {
        StringBuilder text = new StringBuilder();
        for (double value : values) {
            if (text.length() > 0) text.append(',');
            text.append(Double.toString(value));
        }
        return text.toString();
    }
    private static void taskLog(Run run, Chunk chunk, Node node, long compute, long started, String status) {
        System.out.println("TASK|"+run.clientId+"|"+run.id+"|"+chunk.taskId+"|"+node.endpoint+"|"
            +node.language+"|"+chunk.op+"|"+compute+"|"+Math.max(0, System.nanoTime()-started)+"|"+status);
    }
    private void tensor(Run run, String name, int rows, int cols, double[] values) {
        if (traceValues) System.out.println("TENSOR|"+run.clientId+"|"+run.id+"|"+name+"|"+rows+"|"+cols+"|"+csv(values));
    }
    private static int property(String name, int fallback, int min, int max) {
        int result = Integer.getInteger("openaicubes."+name, fallback);
        if (result < min || result > max) throw new IllegalArgumentException("invalid_"+name);
        return result;
    }

    private Model weights() throws Failure {
        if (model == null) synchronized (this) {
            if (model == null) {
                try { model = Model.load(weightsPath()); }
                catch (Exception ex) { throw new Failure("model_unavailable"); }
            }
        }
        return model;
    }
    private static Path weightsPath() throws Exception {
        String configured = System.getProperty("openaicubes.weights");
        if (configured != null) return Paths.get(configured);
        Path path = Paths.get(DistributedCnn.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        for (int i = 0; path != null && i < 10; i++, path = path.getParent()) {
            Path candidate = path.resolve("model/weights/weights.txt");
            if (Files.isRegularFile(candidate)) return candidate;
        }
        return Paths.get("model/weights/weights.txt");
    }
    private static final class Model {
        final double[][] filters;
        final double[] w1, b1, w2, b2;
        Model(Map<String, double[]> values) {
            filters = new double[][] {values.get("FILTER1"), values.get("FILTER2")};
            w1 = values.get("W1"); b1 = values.get("B1"); w2 = values.get("W2"); b2 = values.get("B2");
        }
        static Model load(Path path) throws Exception {
            List<String> lines = new ArrayList<>();
            for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) if (!line.trim().isEmpty()) lines.add(line.trim());
            if (lines.size() != 12) throw new IOException("weight_sections");
            String[] names = {"FILTER1", "FILTER2", "W1", "B1", "W2", "B2"};
            int[] counts = {9, 9, 784, 8, 24, 3};
            Map<String, double[]> values = new HashMap<>();
            for (int i = 0; i < 6; i++) {
                int at = Arrays.asList(names).indexOf(lines.get(2*i));
                if (at < 0 || values.containsKey(names[at])) throw new IOException("weight_sections");
                values.put(names[at], numbers(lines.get(2*i+1), counts[at]));
            }
            return new Model(values);
        }
    }

    @Override public void close() {
        closed.set(true);
        for (Run run : active) run.cancel();
        tasks.shutdownNow();
        deadlines.shutdownNow();
    }
}
