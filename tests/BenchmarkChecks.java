import benchmarks.LocalCnn;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public final class BenchmarkChecks {
    public static void main(String[] args) throws Exception {
        try (LocalCnn sequential = new LocalCnn(Paths.get(args[0]), 1);
             LocalCnn parallel = new LocalCnn(Paths.get(args[0]), 2);
             BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, "UTF-8"))) {
            for (double[] invalid : Arrays.asList(new double[255], new double[257], invalid(Double.NaN), invalid(-1), invalid(Double.POSITIVE_INFINITY), invalid(1.1))) {
                try { sequential.predict(invalid); throw new AssertionError("invalid_image_accepted"); } catch (IllegalArgumentException expected) { }
            }
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",", -1); double[] pixels = new double[tokens.length];
                for (int i = 0; i < pixels.length; i++) pixels[i] = Double.parseDouble(tokens[i]);
                double[] first = sequential.predict(pixels), second = parallel.predict(pixels);
                for (int i = 0; i < 3; i++) if (Math.abs(first[i]-second[i]) > 1e-12) throw new AssertionError("parallel_mismatch");
                System.out.println(first[0]+","+first[1]+","+first[2]);
            }
            if (sequential.peakComputeTasks() != 1 || sequential.computeThreadIds().contains(",")) throw new AssertionError("sequential_threads");
            if (parallel.peakComputeTasks() > 2 || !parallel.computeThreadIds().contains(",")) throw new AssertionError("parallel_threads");
            System.err.println("THREADS sequential="+sequential.computeThreadIds()+" parallel="+parallel.computeThreadIds()+" peak="+parallel.peakComputeTasks());
        }
    }
    private static double[] invalid(double value) { double[] pixels = new double[256]; pixels[13] = value; return pixels; }
}
