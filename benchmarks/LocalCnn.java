package benchmarks;

import java.io.*;
import java.lang.management.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public final class LocalCnn implements AutoCloseable {
    public static final class Prediction {
        public final double[] probabilities;
        public final long cpuNs;
        public final long mathWallNs;
        Prediction(double[] probabilities, long cpuNs, long mathWallNs) { this.probabilities = probabilities; this.cpuNs = cpuNs; this.mathWallNs = mathWallNs; }
    }
    private static final class Accounting {
        final AtomicLong cpu = new AtomicLong(), wall = new AtomicLong();
    }
    private final double[][] filters = new double[2][];
    private final double[] w1, b1, w2, b2;
    private final ExecutorService pool;
    private final int threads;
    private final AtomicInteger active = new AtomicInteger(), peak = new AtomicInteger();
    private final Set<Long> computeIds = ConcurrentHashMap.newKeySet();
    private final ConcurrentMap<Long, AtomicLong> cpuNs = new ConcurrentHashMap<>();
    private final ThreadMXBean bean = ManagementFactory.getThreadMXBean();
    public LocalCnn(Path path, int threads) throws IOException {
        if (threads < 1 || threads > 64) throw new IllegalArgumentException("invalid_compute_threads");
        List<String> lines = new ArrayList<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) if (!line.trim().isEmpty()) lines.add(line.trim());
        String[] keys = {"FILTER1", "FILTER2", "W1", "B1", "W2", "B2"};
        int[] counts = {9, 9, 784, 8, 24, 3};
        if (lines.size() != 12) throw new IOException("weight_sections");
        Map<String, double[]> weights = new HashMap<>();
        for (int i = 0; i < 12; i += 2) {
            int at = Arrays.asList(keys).indexOf(lines.get(i));
            if (at < 0 || weights.containsKey(keys[at])) throw new IOException("weight_sections");
            String[] tokens = lines.get(i+1).split(",", -1);
            if (tokens.length != counts[at]) throw new IOException("weight_dimensions");
            double[] values = new double[tokens.length];
            for (int j = 0; j < values.length; j++) {
                values[j] = Double.parseDouble(tokens[j]);
                if (!Double.isFinite(values[j])) throw new IOException("nonfinite_weight");
            }
            weights.put(keys[at], values);
        }
        filters[0] = weights.get("FILTER1"); filters[1] = weights.get("FILTER2");
        w1 = weights.get("W1"); b1 = weights.get("B1"); w2 = weights.get("W2"); b2 = weights.get("B2");
        this.threads = threads;
        pool = threads == 1 ? null : Executors.newFixedThreadPool(threads, task -> new Thread(task, "cnn-cpu"));
        if (bean.isThreadCpuTimeSupported() && !bean.isThreadCpuTimeEnabled()) bean.setThreadCpuTimeEnabled(true);
    }
    public double[] predict(double[] image) throws Exception {
        return measured(image).probabilities;
    }
    public Prediction measured(double[] image) throws Exception {
        Accounting requestCpu = new Accounting();
        if (image.length != 256) throw new IllegalArgumentException("image_dimensions");
        for (double value : image) if (!Double.isFinite(value) || value < 0 || value > 1) throw new IllegalArgumentException("invalid_pixels");
        double[] flat = new double[98];
        layer(2, requestCpu, i -> {
            double[] conv = new double[196];
            for (int r = 0; r < 14; r++) for (int c = 0; c < 14; c++) {
                double total = 0;
                for (int y = 0; y < 3; y++) for (int x = 0; x < 3; x++) total += image[(r+y)*16+c+x]*filters[i][y*3+x];
                conv[r*14+c] = finite(total);
            }
            for (int r = 0; r < 7; r++) for (int c = 0; c < 7; c++) {
                int at = 2*r*14+2*c;
                flat[i*49+r*7+c] = Math.max(0, Math.max(Math.max(conv[at], conv[at+1]), Math.max(conv[at+14], conv[at+15])));
            }
        });
        double[] hidden = new double[8], logits = new double[3];
        layer(8, requestCpu, i -> { double sum = 0; for (int j = 0; j < 98; j++) sum += w1[i*98+j]*flat[j]; hidden[i] = Math.max(0, finite(sum+b1[i])); });
        layer(3, requestCpu, i -> { double sum = 0; for (int j = 0; j < 8; j++) sum += w2[i*8+j]*hidden[j]; logits[i] = finite(sum+b2[i]); });
        layer(1, requestCpu, ignored -> {
            double max = Math.max(logits[0], Math.max(logits[1], logits[2])), total = 0;
            for (int i = 0; i < 3; i++) { logits[i] = Math.exp(logits[i]-max); total += logits[i]; }
            for (int i = 0; i < 3; i++) logits[i] /= total;
        });
        return new Prediction(logits, requestCpu.cpu.get(), requestCpu.wall.get());
    }
    private interface Work { void run(int index); }
    private void layer(int count, Accounting requestCpu, Work operation) throws Exception {
        if (pool == null) { tracked(requestCpu, () -> { for (int i = 0; i < count; i++) operation.run(i); }); return; }
        List<Callable<Void>> jobs = new ArrayList<>();
        for (int i = 0; i < count; i++) { final int at = i; jobs.add(() -> { tracked(requestCpu, () -> operation.run(at)); return null; }); }
        for (Future<Void> result : pool.invokeAll(jobs)) result.get();
    }
    private void tracked(Accounting requestCpu, Runnable work) {
        long id = Thread.currentThread().getId(), start = cpu();
        computeIds.add(id); int now = active.incrementAndGet(); peak.accumulateAndGet(now, Math::max);
        long wall = System.nanoTime();
        try { work.run(); } finally {
            requestCpu.wall.addAndGet(System.nanoTime()-wall);
            active.decrementAndGet(); long elapsed = Math.max(0, cpu()-start);
            cpuNs.computeIfAbsent(id, key -> new AtomicLong()).addAndGet(elapsed); requestCpu.cpu.addAndGet(elapsed);
        }
    }
    private long cpu() { return bean.isCurrentThreadCpuTimeSupported() ? bean.getCurrentThreadCpuTime() : 0; }
    private static double finite(double value) { if (!Double.isFinite(value)) throw new IllegalArgumentException("nonfinite_result"); return value; }
    public String computeThreadIds() { List<Long> ids = new ArrayList<>(computeIds); Collections.sort(ids); return ids.toString(); }
    public int peakComputeTasks() { return peak.get(); }
    public long computeCpuNs() { long total = 0; for (AtomicLong value : cpuNs.values()) total += value.get(); return total; }
    public int configuredThreads() { return threads; }
    @Override public void close() throws InterruptedException { if (pool != null) { pool.shutdownNow(); if (!pool.awaitTermination(5, TimeUnit.SECONDS)) throw new IllegalStateException("compute_pool_not_stopped"); } }
}
