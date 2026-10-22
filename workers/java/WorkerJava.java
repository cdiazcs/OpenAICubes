import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.*;
import java.util.regex.Pattern;


public class WorkerJava {
    private static final int MAX_LENGTH = 2_000_000;
    private static int computeThreads = 4;
    private static final Pattern DECIMAL = Pattern.compile("[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?");

    public static void main(String[] args) throws Exception {
        int port = args.length >= 1 ? Integer.parseInt(args[0]) : 9101;
        int nThreads = args.length >= 2 ? Integer.parseInt(args[1]) : 4;
        if (nThreads < 1 || nThreads > 256) throw new IllegalArgumentException("invalid_threads");
        computeThreads = nThreads;
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
        if (p[0].equals("CAPS")) {
            if (p.length != 2 || !validId(p[1])) return "ERROR|unknown|invalid_request_id";
            return "CAPS|" + p[1] + "|Java|CONV,DENSE,POST|" + computeThreads;
        }
        if (p[0].equals("JOB")) return job(p);
        if (p.length == 2 && p[0].equals("PING")) return "PONG|" + p[1] + "|Java";
        if (p.length == 8 && p[0].equals("CONV")) {
            int rows = integer(p[2]), cols = integer(p[3]);
            int kr = integer(p[4]), kc = integer(p[5]);
            if (rows < 1 || cols < 1 || kr < 1 || kc < 1 || kr > rows || kc > cols) throw new IllegalArgumentException("dims");
            double[] img = numbers(p[6], Math.multiplyExact(rows, cols));
            double[] kernel = numbers(p[7], Math.multiplyExact(kr, kc));
            int outRows = rows - kr + 1, outCols = cols - kc + 1;
            double[] result = convolution(img, rows, cols, kernel, kr, kc);
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
                h[i] = Math.max(0, finite(sum));
            }
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < no; i++) {
                double sum = b2[i];
                for (int j = 0; j < nh; j++) sum += w2[i * nh+j] * h[j];
                z[i] = finite(sum);
                max = Math.max(max, sum);
            }
            double total = 0;
            for (int i = 0; i < no; i++) { z[i] = Math.exp(z[i] - max); total += z[i]; }
            for (int i = 0; i < no; i++) z[i] /= total;
            return "RESULT|" + p[1] + "|1|" + no + "|" + csv(z);
        }
        throw new IllegalArgumentException("unsupported_command");
    }

    private static String job(String[] p) {
        String requestId = p.length > 1 && validId(p[1]) ? p[1] : "unknown";
        String taskId = p.length > 2 && validId(p[2]) ? p[2] : "unknown";
        try {
            if (p.length < 3 || !validId(p[1]) || !validId(p[2])) throw new IllegalArgumentException("invalid_job_id");
            if (p.length < 4) throw new IllegalArgumentException("invalid_job_fields");
            double[] result;
            int rows, cols;
            long started;
            if (p[3].equals("CONV")) {
                if (p.length != 10) throw new IllegalArgumentException("invalid_job_fields");
                int ir = dimension(p[4]), ic = dimension(p[5]), kr = dimension(p[6]), kc = dimension(p[7]);
                if (kr > ir || kc > ic) throw new IllegalArgumentException("invalid_dimensions");
                rows = ir-kr+1; cols = ic-kc+1;
                if ((long)rows*cols*kr*kc > 10_000_000) throw new IllegalArgumentException("invalid_dimensions");
                double[] image = jobNumbers(p[8], product(ir, ic));
                double[] kernel = jobNumbers(p[9], product(kr, kc));
                started = System.nanoTime();
                result = convolution(image, ir, ic, kernel, kr, kc);
            } else if (p[3].equals("POST")) {
                if (p.length != 8) throw new IllegalArgumentException("invalid_job_fields");
                int ir = dimension(p[5]), ic = dimension(p[6]);
                double[] x = jobNumbers(p[7], product(ir, ic));
                rows = ir; cols = ic; started = System.nanoTime();
                if (p[4].equals("POOL")) {
                    if (ir < 2 || ic < 2) throw new IllegalArgumentException("invalid_dimensions");
                    rows = ir/2; cols = ic/2; result = new double[rows*cols];
                    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
                        int at = 2*r*ic+2*c;
                        result[r*cols+c] = Math.max(0, Math.max(Math.max(x[at], x[at+1]), Math.max(x[at+ic], x[at+ic+1])));
                    }
                } else if (p[4].equals("RELU")) {
                    result = x; for (int i = 0; i < x.length; i++) result[i] = Math.max(0, x[i]);
                } else if (p[4].equals("SOFTMAX")) {
                    if (ir != 1) throw new IllegalArgumentException("invalid_dimensions");
                    double max = x[0], total = 0;
                    for (double value : x) max = Math.max(max, value);
                    result = x; for (int i = 0; i < x.length; i++) { result[i] = Math.exp(x[i]-max); total += result[i]; }
                    for (int i = 0; i < x.length; i++) result[i] /= total;
                } else throw new IllegalArgumentException("unsupported_operation");
            } else if (p[3].equals("DENSE")) {
                if (p.length != 9) throw new IllegalArgumentException("invalid_job_fields");
                int ni = dimension(p[4]);
                rows = 1; cols = dimension(p[5]);
                int weightCount = product(ni, cols);
                double[] x = jobNumbers(p[6], ni), weights = jobNumbers(p[7], weightCount);
                double[] bias = jobNumbers(p[8], cols);
                started = System.nanoTime();
                result = new double[cols];
                for (int i = 0; i < cols; i++) {
                    double sum = 0;
                    for (int j = 0; j < ni; j++) sum += weights[i*ni+j]*x[j];
                    result[i] = finite(sum + bias[i]);
                }
            } else throw new IllegalArgumentException("unsupported_operation");
            long elapsed = Math.max(0, System.nanoTime()-started);
            StringBuilder values = new StringBuilder();
            for (double value : result) {
                if (values.length() > 0) values.append(',');
                values.append(Double.toString(value));
            }
            return "PART|" + requestId + "|" + taskId + "|" + rows + "|" + cols + "|" + values + "|" + elapsed;
        } catch (RuntimeException ex) {
            String code = ex.getMessage();
            if (!java.util.Arrays.asList("invalid_job_id", "invalid_job_fields", "invalid_dimensions",
                    "invalid_numbers", "nonfinite_result", "unsupported_operation").contains(code)) code = "invalid_job";
            return "ERROR|" + requestId + "|" + taskId + "|" + code;
        }
    }

    private static int dimension(String raw) {
        if (!raw.matches("[0-9]{1,7}")) throw new IllegalArgumentException("invalid_dimensions");
        int value = Integer.parseInt(raw);
        if (value < 1 || value > 1_000_000) throw new IllegalArgumentException("invalid_dimensions");
        return value;
    }
    private static int product(int a, int b) {
        if ((long)a*b > 1_000_000) throw new IllegalArgumentException("invalid_dimensions");
        return a*b;
    }
    private static double[] jobNumbers(String raw, int count) {
        String[] tokens = raw.split(",", -1);
        if (tokens.length != count) throw new IllegalArgumentException("invalid_numbers");
        double[] values = new double[count];
        for (int i = 0; i < count; i++) {
            if (tokens[i].length() > 64 || !DECIMAL.matcher(tokens[i]).matches())
                throw new IllegalArgumentException("invalid_numbers");
            values[i] = Double.parseDouble(tokens[i]);
            if (!Double.isFinite(values[i])) throw new IllegalArgumentException("invalid_numbers");
        }
        return values;
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
    private static double[] convolution(double[] image, int rows, int cols, double[] kernel, int kr, int kc) {
        int orows = rows-kr+1, ocols = cols-kc+1;
        double[] result = new double[orows*ocols];
        for (int r = 0; r < orows; r++) for (int c = 0; c < ocols; c++) {
            double sum = 0;
            for (int i = 0; i < kr; i++) for (int j = 0; j < kc; j++)
                sum += image[(r+i)*cols+c+j]*kernel[i*kc+j];
            result[r*ocols+c] = finite(sum);
        }
        return result;
    }

    private static int integer(String s) { return Integer.parseInt(s); }
    private static double finite(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value))
            throw new IllegalArgumentException("nonfinite_result");
        return value;
    }
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
        return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(buf)).toString();
    }
    private static void writeFrame(DataOutputStream out, String message) throws IOException {
        byte[] body = message.getBytes(StandardCharsets.UTF_8);
        if (body.length > MAX_LENGTH) throw new IOException("response_too_long");
        out.writeInt(body.length);
        out.write(body);
        out.flush();
    }
}
