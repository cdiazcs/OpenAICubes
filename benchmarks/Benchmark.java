package benchmarks;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/** Generador de carga OpenAICubes v0.1. Java 8+, solo bibliotecas estandar.
 *  El parametro 'modo' es una etiqueta: NO cambia la configuracion del cluster.
 */
public class Benchmark {
    private static final int MAX_FRAME = 2_000_000;
    private static final String INPUT;
    static {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 256; i++) {
            if (i != 0) sb.append(',');
            sb.append(((i / 16 + i % 16) % 5 == 0) ? '1' : '0');
        }
        INPUT = sb.toString();
    }

    static class Result {
        final int id, client;
        final double ms;
        final boolean ok;
        final String detail;
        Result(int id, int client, double ms, boolean ok, String detail) {
            this.id = id; this.client = client; this.ms = ms;
            this.ok = ok; this.detail = detail;
        }
    }
    private static void send(DataOutputStream out, String text) throws IOException {
        byte[] p = text.getBytes(StandardCharsets.UTF_8);
        if (p.length == 0 || p.length > MAX_FRAME) throw new IOException("request_length");
        out.writeInt(p.length);
        out.write(p);
        out.flush();
    }
    private static String receive(DataInputStream in) throws IOException {
        int len = in.readInt();
        if (len <= 0 || len > MAX_FRAME) throw new IOException("response_length");
        byte[] data = new byte[len];
        in.readFully(data);
        return new String(data, StandardCharsets.UTF_8);
    }
    private static String quoteCsv(String s) { return "\"" + s.replace("\"", "\"\"") + "\""; }
    private static void help() {
        System.err.println("Uso: java -cp build Benchmark <host> <port> <solicitudes> <clientes> <delayMinMs> <delayMaxMs> <modo> <csv> [semilla]");
        System.err.println("Ejemplo: java -cp build Benchmark 127.0.0.1 9000 100 8 1 20 distribuido benchmarks/results/distribuido.csv 42");
        System.err.println("El modo solamente etiqueta resultados; configura aparte los procesos/nucleos/nodos del experimento.");
    }
    public static void main(String[] args) throws Exception {
        if (args.length < 8 || args.length > 9) { help(); return; }
        String host = args[0], mode = args[6];
        int port = Integer.parseInt(args[1]);
        int count = Integer.parseInt(args[2]);
        int clients = Integer.parseInt(args[3]);
        int minDelay = Integer.parseInt(args[4]), maxDelay = Integer.parseInt(args[5]);
        Path csv = Paths.get(args[7]);
        long seed = args.length == 9 ? Long.parseLong(args[8]) : 42L;
        if (count < 1 || clients < 1 || clients > 1000 || minDelay < 0 || maxDelay < minDelay || maxDelay > 60000) {
            throw new IllegalArgumentException("Parametros invalidos");
        }
        ExecutorService pool = Executors.newFixedThreadPool(clients);
        CountDownLatch start = new CountDownLatch(1);
        Queue<Result> records = new ConcurrentLinkedQueue<>();
        List<Future<?>> futures = new ArrayList<>();
        long experimentStart = System.nanoTime();
        for (int w = 0; w < clients; w++) {
            final int clientId = w;
            futures.add(pool.submit(() -> {
                Random random = new Random(seed + 10007L * clientId);
                try (Socket socket = new Socket()) {
                    socket.connect(new InetSocketAddress(host, port), 5000);
                    socket.setSoTimeout(15000);
                    DataInputStream in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
                    DataOutputStream out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
                    start.await();
                    for (int id = clientId; id < count; id += clients) {
                        int delay = minDelay + (maxDelay > minDelay ? random.nextInt(maxDelay - minDelay + 1) : 0);
                        if (delay > 0) Thread.sleep(delay);
                        String requestId = "b" + id;
                        String request = "CONV|" + requestId + "|16|16|3|3|" + INPUT + "|1,0,-1,1,0,-1,1,0,-1";
                        long t0 = System.nanoTime();
                        try {
                            send(out, request);
                            String response = receive(in);
                            double elapsedMs = (System.nanoTime() - t0) / 1_000_000.0;
                            boolean ok = response.startsWith("RESULT|" + requestId + "|14|14|");
                            records.add(new Result(id, clientId, elapsedMs, ok, ok ? "ok" : response));
                        } catch (IOException ex) {
                            double elapsedMs = (System.nanoTime() - t0) / 1_000_000.0;
                            records.add(new Result(id, clientId, elapsedMs, false, ex.toString()));
                            // Conexión rota: las restantes solicitudes de este cliente no se enviaron.
                            for (int rest = id + clients; rest < count; rest += clients)
                                records.add(new Result(rest, clientId, 0, false, "not_sent_connection_lost"));
                            break;
                        }
                    }
                } catch (Exception ex) {
                    for (int id = clientId; id < count; id += clients)
                        records.add(new Result(id, clientId, 0, false, "not_sent_" + ex.toString()));
                }
            }));
        }
        start.countDown();
        for (Future<?> f : futures) f.get();
        pool.shutdown();
        double totalSecs = (System.nanoTime() - experimentStart) / 1_000_000_000.0;
        List<Result> rows = new ArrayList<>(records);
        rows.sort(Comparator.comparingInt(r -> r.id));
        if (csv.getParent() != null) Files.createDirectories(csv.getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(csv, StandardCharsets.UTF_8)) {
            writer.write("modo,requestId,cliente,latencia_ms,ok,detalle\n");
            for (Result r : rows) {
                writer.write(quoteCsv(mode) + ",b" + r.id + "," + r.client + "," +
                    String.format(Locale.US,"%.6f",r.ms) + "," + (r.ok ? 1 : 0) + "," + quoteCsv(r.detail) + "\n");
            }
        }
        List<Double> okMs = new ArrayList<>();
        for (Result r : rows) if (r.ok) okMs.add(r.ms);
        Collections.sort(okMs);
        long success = okMs.size(), errors = rows.size() - success;
        double mean = 0;
        for (double x : okMs) mean += x;
        if (success > 0) mean /= success;
        double p95 = success > 0 ? okMs.get(Math.max(0,(int)Math.ceil(success * 0.95) - 1)) : Double.NaN;
        System.out.println("Modo: " + mode + " | solicitudes=" + rows.size() + " | correctas=" + success + " | errores=" + errors);
        System.out.printf(Locale.US, "Duracion=%.3fs | throughput=%.2f req/s | latenciaMedia=%.3fms | p95=%.3fms%n",
            totalSecs, success / totalSecs, mean, p95);
        System.out.println("CSV: " + csv.toAbsolutePath());
    }
}
