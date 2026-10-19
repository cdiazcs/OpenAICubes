import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.*;

/** Worker Java 8+: CONV y MLP usando solo JDK. */
public class WorkerJava {
    private static final int MAX_LENGTH = 2_000_000;

    public static void main(String[] args) throws Exception {
        int port = args.length >= 1 ? Integer.parseInt(args[0]) : 9101;
        int nThreads = args.length >= 2 ? Integer.parseInt(args[1]) : 4;
        ExecutorService pool = Executors.newFixedThreadPool(nThreads);
        try (ServerSocket server = new ServerSocket(port)) {
            System.out.println("WorkerJava escuchando en puerto " + port + " con " + nThreads + " hilos");
            while (true) {
                Socket client = server.accept();
                pool.execute(() -> handle(client));
            }
        } finally {
            pool.shutdown();
        }
    }

    private static void handle(Socket socket) {
        try (Socket client = socket;
             DataInputStream in = new DataInputStream(new BufferedInputStream(client.getInputStream()));
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(client.getOutputStream()))) {
            client.setSoTimeout(15000);
            while (true) {
                String input;
                try { input = readFrame(in); }
                catch (EOFException eof) { return; }
                String answer;
                try { answer = evaluate(input); }
                catch (Exception error) {
                    String[] parts = input.split("\\|", -1);
                    String id = parts.length > 1 ? parts[1] : "unknown";
                    answer = "ERROR|" + id + "|" + error.getClass().getSimpleName();
                }
                writeFrame(out, answer);
            }
        } catch (IOException e) {
            System.err.println("conexion terminada: " + e.getMessage());
        }
    }

    private static String evaluate(String input) {
        String[] p = input.split("\\|", -1);
        if (p.length == 2 && p[0].equals("PING")) return "PONG|" + p[1] + "|Java";
        if (p.length == 8 && p[0].equals("CONV")) {
            int rows = integer(p[2]), cols = integer(p[3]);
            int kr = integer(p[4]), kc = integer(p[5]);
            if (rows < 1 || cols < 1 || kr < 1 || kc < 1 || kr > rows || kc > cols) throw new IllegalArgumentException("dims");
            double[] img = numbers(p[6], rows * cols);
            double[] kernel = numbers(p[7], kr * kc);
            int outRows = rows - kr + 1, outCols = cols - kc + 1;
            double[] result = new double[outRows * outCols];
            for (int r = 0; r < outRows; r++) for (int c = 0; c < outCols; c++) {
                double sum = 0;
                for (int i = 0; i < kr; i++) for (int j = 0; j < kc; j++)
                    sum += img[(r+i) * cols + c+j] * kernel[i * kc + j];
                result[r * outCols + c] = sum;
            }
            return "RESULT|" + p[1] + "|" + outRows + "|" + outCols + "|" + csv(result);
        }
        if (p.length == 10 && p[0].equals("MLP")) {
            int ni = integer(p[2]), nh = integer(p[3]), no = integer(p[4]);
            if (ni < 1 || nh < 1 || no < 1 || (long)ni * nh > 1000000 || (long)nh * no > 1000000)
                throw new IllegalArgumentException("dims");
            double[] x = numbers(p[5], ni), w1 = numbers(p[6], nh * ni), b1 = numbers(p[7], nh);
            double[] w2 = numbers(p[8], no * nh), b2 = numbers(p[9], no);
            double[] h = new double[nh], z = new double[no];
            for (int i = 0; i < nh; i++) {
                double sum = b1[i];
                for (int j = 0; j < ni; j++) sum += w1[i * ni+j] * x[j];
                h[i] = Math.max(0, sum);
            }
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < no; i++) {
                double sum = b2[i];
                for (int j = 0; j < nh; j++) sum += w2[i * nh+j] * h[j];
                z[i] = sum;
                max = Math.max(max, sum);
            }
            double total = 0;
            for (int i = 0; i < no; i++) { z[i] = Math.exp(z[i] - max); total += z[i]; }
            for (int i = 0; i < no; i++) z[i] /= total;
            return "RESULT|" + p[1] + "|1|" + no + "|" + csv(z);
        }
        throw new IllegalArgumentException("unsupported_command");
    }

    private static int integer(String s) { return Integer.parseInt(s); }
    private static double[] numbers(String raw, int count) {
        if (count < 1 || count > 1000000) throw new IllegalArgumentException("array_size");
        String[] t = raw.split(",", -1);
        if (t.length != count) throw new IllegalArgumentException("array_length");
        double[] d = new double[count];
        for (int i = 0; i < count; i++) {
            d[i] = Double.parseDouble(t[i]);
            if (Double.isNaN(d[i]) || Double.isInfinite(d[i])) throw new IllegalArgumentException("nonfinite");
        }
        return d;
    }
    private static String csv(double[] values) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format(Locale.ROOT, "%.12g", values[i]));
        }
        return sb.toString();
    }
    private static String readFrame(DataInputStream in) throws IOException {
        int length = in.readInt();
        if (length <= 0 || length > MAX_LENGTH) throw new IOException("invalid_length");
        byte[] buf = new byte[length];
        in.readFully(buf);
        return new String(buf, StandardCharsets.UTF_8);
    }
    private static void writeFrame(DataOutputStream out, String message) throws IOException {
        byte[] body = message.getBytes(StandardCharsets.UTF_8);
        if (body.length > MAX_LENGTH) throw new IOException("response_too_long");
        out.writeInt(body.length);
        out.write(body);
        out.flush();
    }
}
