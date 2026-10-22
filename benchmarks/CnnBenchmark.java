package benchmarks;

import java.io.*;
import java.lang.management.ManagementFactory;
import java.lang.reflect.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;

public final class CnnBenchmark {
    private static final class Image {
        String name, hash, csv;
        double[] pixels, expected;
    }
    private static final class Row {
        int index, client, delay;
        Image image;
        String id, cause = "", detail = "";
        boolean ok;
        double ms, cpuMs = Double.NaN, mathWallMs = Double.NaN;
        double[] probabilities;
    }
    private static final class Failure extends IOException {
        final String cause;
        Failure(String cause, String detail) { super(detail); this.cause = cause; }
    }
    public static void main(String[] args) {
        try { run(configuration(args)); }
        catch (Exception error) { error.printStackTrace(System.err); System.exit(2); }
    }
    private static Properties configuration(String[] args) throws Exception {
        Properties p = new Properties();
        p.setProperty("images", "model/examples"); p.setProperty("weights", "model/weights/weights.txt");
        p.setProperty("requests", "300"); p.setProperty("clients", "1"); p.setProperty("warmup", "50");
        p.setProperty("seed", "42"); p.setProperty("delay.min.ms", "1"); p.setProperty("delay.max.ms", "3");
        p.setProperty("timeout.ms", "15000"); p.setProperty("cores", "1"); p.setProperty("threads", "1");
        p.setProperty("output", "build/stage6/measurement.csv");
        if (args.length >= 2 && args[0].equals("--config")) {
            try (Reader reader = Files.newBufferedReader(Paths.get(args[1]), StandardCharsets.UTF_8)) { p.load(reader); }
            for (int i = 2; i < args.length; i++) {
                int at = args[i].indexOf('='); if (at < 1) throw new IllegalArgumentException("expected_key=value");
                p.setProperty(args[i].substring(0, at), args[i].substring(at+1));
            }
        } else if (args.length == 8 || args.length == 9) {
            p.setProperty("host", args[0]); p.setProperty("port", args[1]); p.setProperty("requests", args[2]);
            p.setProperty("clients", args[3]); p.setProperty("delay.min.ms", args[4]); p.setProperty("delay.max.ms", args[5]);
            p.setProperty("legacy.label", args[6]); p.setProperty("output", args[7]); p.setProperty("scenario", "SOCKET_CONTROL");
            p.setProperty("warmup", "0"); if (args.length == 9) p.setProperty("seed", args[8]);
        } else throw new IllegalArgumentException("Benchmark --config path [key=value ...] OR host port requests clients minDelay maxDelay label csv [seed]");
        String scenario = p.getProperty("scenario", "");
        if (!Arrays.asList("LOCAL_SEQUENTIAL", "LOCAL_PARALLEL", "DISTRIBUTED", "PARALLEL_DISTRIBUTED", "SOCKET_CONTROL").contains(scenario)) throw new IllegalArgumentException("invalid_scenario");
        int threads = number(p, "threads", 1, 64), cores = number(p, "cores", 1, 64);
        if (scenario.equals("LOCAL_SEQUENTIAL") && (threads != 1 || cores != 1)) throw new IllegalArgumentException("sequential_requires_one_thread_and_core");
        if (scenario.equals("LOCAL_PARALLEL") && (threads < 2 || threads != cores)) throw new IllegalArgumentException("parallel_requires_distinct_physical_cores");
        return p;
    }
    private static int number(Properties p, String key, int min, int max) {
        int value = Integer.parseInt(p.getProperty(key));
        if (value < min || value > max) throw new IllegalArgumentException("invalid_"+key); return value;
    }
    private static String hash(byte[] bytes) throws Exception {
        StringBuilder value = new StringBuilder();
        for (byte b : MessageDigest.getInstance("SHA-256").digest(bytes)) value.append(String.format(Locale.ROOT, "%02x", b & 255));
        return value.toString();
    }
    private static List<Image> images(Properties p) throws Exception {
        List<Path> paths = new ArrayList<>();
        try (DirectoryStream<Path> files = Files.newDirectoryStream(Paths.get(p.getProperty("images")))) {
            for (Path path : files) if (Files.isRegularFile(path) && path.toString().toLowerCase(Locale.ROOT).matches(".*\\.(png|pgm|jpg|jpeg)$")) paths.add(path);
        }
        Collections.sort(paths); if (paths.isEmpty() || paths.size() > 10000) throw new IOException("invalid_image_count");
        Method prepare = Class.forName("ClientCli").getMethod("prepareImage", File.class);
        List<Image> images = new ArrayList<>();
        try (LocalCnn reference = new LocalCnn(Paths.get(p.getProperty("weights")), 1)) {
            for (Path path : paths) {
                Image image = new Image(); image.name = path.getFileName().toString(); image.hash = hash(Files.readAllBytes(path));
                Object prepared = prepare.invoke(null, path.toFile());
                image.pixels = (double[])prepared.getClass().getMethod("pixels").invoke(prepared);
                StringBuilder csv = new StringBuilder(); for (double value : image.pixels) { if (csv.length() > 0) csv.append(','); csv.append(Double.toString(value)); }
                image.csv = csv.toString(); image.expected = reference.predict(image.pixels); images.add(image);
            }
        }
        return images;
    }
    private static LocalCnn.Prediction local(LocalCnn model, ExecutorService sequential, double[] pixels) throws Exception {
        return sequential == null ? model.measured(pixels) : sequential.submit(() -> model.measured(pixels)).get();
    }
    private static double[] response(String reply, String id) throws Failure {
        String[] fields = reply.split("\\|", -1);
        if ((fields.length == 3 || fields.length == 4) && fields[0].equals("ERROR") && fields[1].equals(id)) throw new Failure("processing", fields[2]+(fields.length == 4 ? ":"+fields[3] : ""));
        if (fields.length != 5 || !fields[0].equals("RESULT") || !fields[1].equals(id) || !fields[2].equals("1") || !fields[3].equals("3")) throw new Failure("validation", "response_shape_or_id");
        String[] tokens = fields[4].split(",", -1); if (tokens.length != 3) throw new Failure("validation", "probability_count");
        double[] values = new double[3]; double total = 0;
        try {
            for (int i = 0; i < 3; i++) {
                if (tokens[i].length() > 64 || !tokens[i].matches("[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?")) throw new NumberFormatException();
                values[i] = Double.parseDouble(tokens[i]); if (!Double.isFinite(values[i]) || values[i] < 0 || values[i] > 1) throw new NumberFormatException(); total += values[i];
            }
        } catch (NumberFormatException error) { throw new Failure("validation", "invalid_probability"); }
        if (Math.abs(total-1) > 1e-8) throw new Failure("validation", "probability_sum"); return values;
    }
    private static void execute(Row row, Properties p, LocalCnn model, ExecutorService sequential) {
        long start = System.nanoTime();
        try {
            if (model != null) {
                LocalCnn.Prediction prediction = local(model, sequential, row.image.pixels);
                row.probabilities = prediction.probabilities; row.cpuMs = prediction.cpuNs/1e6; row.mathWallMs = prediction.mathWallNs/1e6;
            } else {
                String mode = p.getProperty("scenario").equals("SOCKET_CONTROL") ? "SEQUENTIAL" : "DISTRIBUTED";
                row.probabilities = response(PhysicalConfig.rpc(p.getProperty("host")+":"+p.getProperty("port"), "IDENTIFY|"+row.id+"|"+row.image.csv+"|"+mode, Integer.parseInt(p.getProperty("timeout.ms"))), row.id);
            }
            for (int i = 0; i < 3; i++) if (Math.abs(row.probabilities[i]-row.image.expected[i]) > 1e-8) throw new Failure("numeric", "probability_mismatch");
            if (maximum(row.probabilities) != maximum(row.image.expected)) throw new Failure("numeric", "class_mismatch");
            row.ok = true;
        } catch (Exception error) {
            Throwable cause = error; while (cause instanceof ExecutionException && cause.getCause() != null) cause = cause.getCause();
            row.cause = cause instanceof Failure ? ((Failure)cause).cause : cause instanceof SocketTimeoutException ? "timeout" :
                cause instanceof ConnectException || cause instanceof UnknownHostException || cause instanceof NoRouteToHostException ? "connection_not_sent" :
                cause instanceof EOFException || cause instanceof SocketException ? "connection_lost" : cause instanceof IOException ? "validation" : "local_processing";
            row.detail = cause.toString();
        } finally { row.ms = (System.nanoTime()-start)/1e6; }
    }
    private static int maximum(double[] values) { int at = 0; for (int i = 1; i < values.length; i++) if (values[i] > values[at]) at = i; return at; }
    private static void run(Properties p) throws Exception {
        int requests = number(p, "requests", 1, 1000000), clients = number(p, "clients", 1, 256), warmup = number(p, "warmup", 0, 1000000);
        int low = number(p, "delay.min.ms", 0, 60000), high = number(p, "delay.max.ms", low, 60000);
        number(p, "timeout.ms", 1, 300000);
        Path output = Paths.get(p.getProperty("output")).toAbsolutePath(); Files.createDirectories(output.getParent());
        p.setProperty("weights.sha256", hash(Files.readAllBytes(Paths.get(p.getProperty("weights")))));
        List<Image> images = images(p); StringBuilder fingerprint = new StringBuilder();
        for (Image image : images) fingerprint.append(image.name).append(':').append(image.hash).append('\n');
        String dataset = hash(fingerprint.toString().getBytes(StandardCharsets.UTF_8));
        Row[] rows = new Row[requests]; Random random = new Random(Long.parseLong(p.getProperty("seed")));
        String token = Long.toHexString(System.currentTimeMillis())+"-"+ManagementFactory.getRuntimeMXBean().getName().split("@")[0];
        StringBuilder load = new StringBuilder(dataset+"|"+p.getProperty("weights.sha256")+"|"+clients+"|"+warmup+"|"+p.getProperty("seed"));
        for (int i = 0; i < requests; i++) {
            Row row = new Row(); row.index = i; row.client = i%clients; row.image = images.get(random.nextInt(images.size())); row.delay = low+random.nextInt(high-low+1);
            row.id = p.containsKey("legacy.label") ? "b"+i : "bench-"+token+"-"+i; rows[i] = row;
            load.append('|').append(row.image.hash).append(':').append(row.delay);
        }
        p.setProperty("dataset.sha256", dataset); p.setProperty("load.sha256", hash(load.toString().getBytes(StandardCharsets.UTF_8)));
        String scenario = p.getProperty("scenario"); boolean isLocal = scenario.startsWith("LOCAL_");
        PhysicalConfig physical = scenario.equals("SOCKET_CONTROL") ? null : new PhysicalConfig(p, output);
        if (physical != null) physical.before();
        LocalCnn model = isLocal ? new LocalCnn(Paths.get(p.getProperty("weights")), Integer.parseInt(p.getProperty("threads"))) : null;
        ExecutorService sequential = scenario.equals("LOCAL_SEQUENTIAL") && clients > 1 ? Executors.newSingleThreadExecutor() : null;
        ExecutorService emitters = clients > 1 ? Executors.newFixedThreadPool(clients) : null;
        long elapsed = 0; String date = Instant.now().toString();
        try {
            for (int i = 0; i < warmup; i++) {
                Row row = new Row(); row.id = "warm-"+token+"-"+i; row.image = images.get(i%images.size()); execute(row, p, model, sequential);
                if (!row.ok) throw new IOException("warmup_failed: "+row.cause+" "+row.detail);
            }
            List<Callable<Void>> work = new ArrayList<>();
            for (int c = 0; c < clients; c++) { final int client = c; work.add(() -> { for (int i = client; i < requests; i += clients) { Row row = rows[i]; if (row.delay > 0) Thread.sleep(row.delay); execute(row, p, model, sequential); } return null; }); }
            long start = System.nanoTime();
            if (emitters == null) work.get(0).call(); else for (Future<Void> task : emitters.invokeAll(work)) task.get();
            elapsed = System.nanoTime()-start;
            boolean valid = physical == null || physical.after();
            Properties summary = summarize(p, rows, elapsed, date);
            if (physical != null) summary.putAll(physical.evidence); else { summary.setProperty("verification", "socket_control_only"); summary.setProperty("configuration_valid", "true"); }
            if (model != null) {
                summary.setProperty("compute.thread.ids", model.computeThreadIds()); summary.setProperty("compute.peak.tasks", Integer.toString(model.peakComputeTasks()));
                double cpu = 0, mathWall = 0; for (Row row : rows) { if (Double.isFinite(row.cpuMs)) cpu += row.cpuMs; if (Double.isFinite(row.mathWallMs)) mathWall += row.mathWallMs; }
                summary.setProperty("pure.cpu.ms", Double.toString(cpu)); summary.setProperty("pure.cpu.scope", "sum_thread_cpu_math_excludes_warmup_queue_network");
                summary.setProperty("pure.cpu.status", cpu == 0 ? "zero_at_os_timer_resolution" : "measured_at_os_timer_resolution");
                summary.setProperty("math.task.wall.ms", Double.toString(mathWall));
                summary.setProperty("math.task.wall.scope", "sum_math_task_wall_time_excludes_dispatch_queue_barriers_network_includes_preemption");
            } else summary.setProperty("pure.cpu.scope", "not_measured_remote_task_wall_time_is_not_cpu_time");
            summary.setProperty("jvm.total.threads.after", Integer.toString(ManagementFactory.getThreadMXBean().getThreadCount()));
            speedup(summary, p);
            csv(output, rows, p);
            summary.setProperty("requests.csv.sha256", hash(Files.readAllBytes(output)));
            try (Writer writer = Files.newBufferedWriter(Paths.get(output+".summary.properties"), StandardCharsets.UTF_8)) { summary.store(writer, null); }
            System.out.println(scenario+" requests="+requests+" success="+summary.getProperty("success")+" errors="+summary.getProperty("errors")+" mean_ms="+summary.getProperty("mean.ms")+" p95_ms="+summary.getProperty("p95.ms")+" throughput="+summary.getProperty("throughput.rps")+" verification="+summary.getProperty("verification")+" output="+output);
            if (!valid) throw new IOException("configuration_changed_during_measurement");
        } finally {
            if (emitters != null) emitters.shutdownNow(); if (sequential != null) sequential.shutdownNow(); if (model != null) model.close();
        }
    }
    private static Properties summarize(Properties p, Row[] rows, long elapsed, String date) {
        Properties s = new Properties(); s.putAll(p); s.setProperty("schema", "openaicubes-benchmark-6"); s.setProperty("operation", "IDENTIFY"); s.setProperty("date.utc", date);
        s.setProperty("java.version", System.getProperty("java.version")); s.setProperty("java.vm", System.getProperty("java.vm.name"));
        s.setProperty("input.scope", "same_predecoded_16x16_pixels_excludes_file_IO_and_image_preprocessing");
        s.setProperty("total.scope", "measured_load_includes_think_delay_excludes_preflight_warmup_and_output");
        s.setProperty("compute.threads.scope", "CNN_math_slots_excludes_JIT_GC_IO_control_threads");
        s.setProperty("speedup.service.scope", "sum_inference_latency_including_queue_and_dispatch");
        List<Double> latency = new ArrayList<>(); Map<String, Integer> errors = new TreeMap<>(); double service = 0; int success = 0;
        for (Row row : rows) { service += row.ms; if (row.ok) { success++; latency.add(row.ms); } else errors.merge(row.cause, 1, Integer::sum); }
        Collections.sort(latency); double total = elapsed/1e9, sum = 0; for (double ms : latency) sum += ms;
        s.setProperty("samples", Integer.toString(rows.length)); s.setProperty("success", Integer.toString(success)); s.setProperty("errors", Integer.toString(rows.length-success));
        for (String cause : Arrays.asList("connection_not_sent", "timeout", "connection_lost", "processing", "validation", "numeric", "local_processing")) s.setProperty("errors."+cause, Integer.toString(errors.getOrDefault(cause, 0)));
        s.setProperty("total.s", Double.toString(total)); s.setProperty("service.s", Double.toString(service/1000));
        s.setProperty("mean.ms", latency.isEmpty() ? "" : Double.toString(sum/latency.size()));
        s.setProperty("p95.ms", latency.isEmpty() ? "" : Double.toString(latency.get((int)Math.ceil(.95*latency.size())-1)));
        s.setProperty("latency.scope", "successful_requests_includes_queue_connection_validation_excludes_think_delay");
        s.setProperty("throughput.rps", Double.toString(success/total)); s.setProperty("numeric.tolerance", "1e-8");
        return s;
    }
    private static void speedup(Properties s, Properties p) throws Exception {
        s.setProperty("speedup.total", ""); s.setProperty("speedup.service", "");
        if (!s.getProperty("errors").equals("0") || !s.getProperty("configuration_valid").equals("true") || s.getProperty("verification").equals("localhost_emulation")) return;
        if (s.getProperty("scenario").equals("LOCAL_SEQUENTIAL")) { s.setProperty("speedup.total", "1.0"); s.setProperty("speedup.service", "1.0"); return; }
        if (!p.containsKey("baseline") || !Files.isRegularFile(Paths.get(p.getProperty("baseline")))) return;
        Properties base = new Properties(); try (Reader reader = Files.newBufferedReader(Paths.get(p.getProperty("baseline")), StandardCharsets.UTF_8)) { base.load(reader); }
        if (!base.getProperty("scenario", "").equals("LOCAL_SEQUENTIAL") || !base.getProperty("load.sha256", "").equals(s.getProperty("load.sha256")) || !base.getProperty("errors", "").equals("0") || !base.getProperty("configuration_valid", "").equals("true")) return;
        s.setProperty("speedup.total", Double.toString(Double.parseDouble(base.getProperty("total.s"))/Double.parseDouble(s.getProperty("total.s"))));
        s.setProperty("speedup.service", Double.toString(Double.parseDouble(base.getProperty("service.s"))/Double.parseDouble(s.getProperty("service.s"))));
    }
    private static String quote(String value) { return "\""+value.replace("\"", "\"\"").replace('\r', ' ').replace('\n', ' ')+"\""; }
    private static void csv(Path path, Row[] rows, Properties p) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("modo,requestId,cliente,latencia_ms,ok,detalle,imagen,imagen_sha256,retraso_ms,error_causa,p0,p1,p2,cpu_puro_ms,matematica_tareas_ms\n");
            for (Row row : rows) {
                writer.write(quote(p.getProperty("legacy.label", p.getProperty("scenario")))+","+quote(row.id)+","+row.client+","+row.ms+","+(row.ok ? "1" : "0")+","+quote(row.detail)+","+quote(row.image.name)+","+row.image.hash+","+row.delay+","+row.cause);
                for (int i = 0; i < 3; i++) writer.write(","+(row.probabilities == null ? "" : Double.toString(row.probabilities[i])));
                writer.write(","+(Double.isFinite(row.cpuMs) ? Double.toString(row.cpuMs) : "")+","+(Double.isFinite(row.mathWallMs) ? Double.toString(row.mathWallMs) : "")+"\n");
            }
        }
    }
}
