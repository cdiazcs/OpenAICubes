import java.io.*;
import java.net.*;
import java.awt.color.ColorSpace;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;
import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import javax.imageio.stream.MemoryCacheImageInputStream;


public class ClientCli {
    private static final int MAX_BYTES = 2_000_000;
    private static final int MAX_PIXELS = 4_194_304;
    private static final String[] CLASSES = {"circulo", "cruz", "triangulo"};
    private static final Pattern DECIMAL = Pattern.compile("[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?");

    public static void main(String[] args) {
        try { run(args); }
        catch (Exception ex) {
            System.err.println("ERROR: " + ex.getMessage());
            System.exit(1);
        }
    }

    private static void run(String[] args) throws Exception {
        // Si faltan datos, usa el servidor local y el puerto por defecto.
        String host = args.length > 0 ? args[0] : "127.0.0.1";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 9000;
        boolean identify = args.length > 2;
        int networkTimeout = Integer.getInteger("openaicubes.clientTimeoutMs", 15000);
        if (networkTimeout < 1 || networkTimeout > 120000) throw new IllegalArgumentException("invalid_timeout");
        String requestId = null;
        String mode = args.length == 6 ? args[5].toUpperCase(Locale.ROOT) : "SEQUENTIAL";
        if (!mode.equals("SEQUENTIAL") && !mode.equals("DISTRIBUTED")) throw new IllegalArgumentException("invalid_mode");
        if (identify && (args.length < 4 || args.length > 6 || !args[2].equalsIgnoreCase("IDENTIFY")))
            throw new IllegalArgumentException("Uso: java -cp build ClientCli [host puerto [IDENTIFY imagen.png|imagen.pgm [requestId [SEQUENTIAL|DISTRIBUTED]]]]");
        String[] cases = {
            "PING|demo0",
            "CONV|demo1|3|3|2|2|1,2,3,4,5,6,7,8,9|1,0,0,-1",
            "MLP|demo2|2|2|2|1,2|1,0,0,1|0,0|1,0,0,1|0,0"
        };
        if (identify) {
            // Convierte la imagen en una lista de valores para enviarla al servidor.
            requestId = args.length >= 5 ? args[4] : UUID.randomUUID().toString();
            if (!validId(requestId)) throw new IllegalArgumentException("invalid_request_id");
            double[] pixels = readImage(new File(args[3]));
            StringBuilder csv = new StringBuilder();
            for (double value : pixels) {
                if (csv.length() > 0) csv.append(',');
                csv.append(Double.toString(value));
            }
            cases = new String[] {"IDENTIFY|" + requestId + "|" + csv + (args.length == 6 ? "|" + mode : "")};
        }
        Timer deadlines = new Timer("client-network-deadline", true);
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(host, port), Math.min(5000, networkTimeout));
            try (
             DataOutputStream out = new DataOutputStream(socket.getOutputStream());
             DataInputStream in = new DataInputStream(socket.getInputStream())) {
            socket.setSoTimeout(networkTimeout);
            for (String s : cases) {
                TimerTask guard = new TimerTask() {
                    // Cierra la conexión si el servidor supera el tiempo límite.
                    public void run() { try { socket.close(); } catch (IOException ignored) { } }
                };
                deadlines.schedule(guard, networkTimeout);
                try {
                    long started = System.nanoTime();
                    String reply = exchange(in, out, s, () -> { });
                    if (identify) {
                        showPrediction(reply, requestId, (System.nanoTime()-started)/1_000_000.0);
                        System.out.println("Modo: " + mode);
                    }
                    else System.out.println(s + "\n  => " + reply + "\n");
                } finally { guard.cancel(); }
            }
            }
        } finally { deadlines.cancel(); }
    }

  
    public static String exchange(DataInputStream in, DataOutputStream out, String request,
                                  Runnable sent) throws IOException {
        byte[] body = request.getBytes(StandardCharsets.UTF_8);
        if (body.length == 0 || body.length > MAX_BYTES) throw new IOException("invalid_request_length");
        // Cada mensaje empieza con su tamaño para que el receptor sepa cuánto leer.
        out.writeInt(body.length); out.write(body); out.flush();
        sent.run();
        int length = in.readInt();
        if (length <= 0 || length > MAX_BYTES) throw new IOException("invalid_response_length");
        byte[] response = new byte[length];
        in.readFully(response);
        return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(response)).toString();
    }

    public static final class Prediction {
        public final String reply;
        public final int index;
        private final double[] probabilities;
        private Prediction(String reply, int index, double[] probabilities) {
            this.reply = reply; this.index = index; this.probabilities = probabilities.clone();
        }
        public String label() { return CLASSES[index]; }
        public double confidence() { return probabilities[index]; }
        public double[] probabilities() { return probabilities.clone(); }
    }

    public static final class ServerError extends IOException {
        public final String code;
        private ServerError(String code) { super(code); this.code = code; }
    }

    private static void showPrediction(String reply, String id, double millis) throws IOException {
        Prediction prediction = parsePrediction(reply, id);
        double[] probabilities = prediction.probabilities();
        System.out.println(reply);
        System.out.println("Clase: " + prediction.label() + " (indice=" + prediction.index + ")");
        System.out.printf(Locale.ROOT, "Probabilidades: circulo=%.12g cruz=%.12g triangulo=%.12g%n",
                          probabilities[0], probabilities[1], probabilities[2]);
        System.out.printf(Locale.ROOT, "Tiempo de respuesta: %.3f ms%n", millis);
    }

    
    public static Prediction parsePrediction(String reply, String id) throws IOException {
        if (reply == null || !validId(id)) throw new IOException("invalid_identify_response");
        String[] fields = reply.split("\\|", -1);
        if (fields.length == 3 && fields[0].equals("ERROR") && fields[1].equals(id)
                && fields[2].matches("[A-Za-z0-9_:-]{1,128}")) throw new ServerError(fields[2]);
        if (fields.length != 5 || !fields[0].equals("RESULT") || !fields[1].equals(id)
                || !fields[2].equals("1") || !fields[3].equals("3")) throw new IOException("invalid_identify_response");
        String[] tokens = fields[4].split(",", -1);
        if (tokens.length != 3) throw new IOException("invalid_probabilities");
        double[] probabilities = new double[3];
        double sum = 0;
        int best = 0;
        // Revisa cada probabilidad y guarda la clase con el valor más alto.
        for (int i = 0; i < 3; i++) {
            if (tokens[i].length() > 64 || !DECIMAL.matcher(tokens[i]).matches()) throw new IOException("invalid_probabilities");
            double value = Double.parseDouble(tokens[i]);
            if (!Double.isFinite(value) || value < 0 || value > 1) throw new IOException("invalid_probabilities");
            probabilities[i] = value;
            sum += value;
            if (value > probabilities[best]) best = i;
        }
        if (Math.abs(sum-1) > 1e-8) throw new IOException("invalid_probabilities");
        return new Prediction(reply, best, probabilities);
    }

    public static boolean validId(String id) {
       
        if (id == null || id.isEmpty() || id.codePointCount(0, id.length()) > 128) return false;
        for (int at = 0; at < id.length();) {
            int cp = id.codePointAt(at);
            if (cp == '|' || Character.isISOControl(cp) || Character.isWhitespace(cp)
                    || Character.isSpaceChar(cp) || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
            at += Character.charCount(cp);
        }
        return true;
    }

   
    public static final class PreparedImage {
        public final String name;
        public final int width, height;
        private final double[] pixels;
        private final BufferedImage preview;
        private PreparedImage(File file, int width, int height, double[] pixels, BufferedImage preview) {
            this.name = file.getName(); this.width = width; this.height = height;
            this.pixels = pixels.clone(); this.preview = thumbnail(preview, 480);
        }
        public double[] pixels() { return pixels.clone(); }
        public BufferedImage preview() { return thumbnail(preview, 480); }
    }

    public static BufferedImage thumbnail(BufferedImage image, int maximum) {
        if (maximum < 1 || maximum > 480) throw new IllegalArgumentException("invalid_thumbnail_size");
        // Reduce la imagen sin cambiar su proporción ni ampliarla.
        double scale = Math.min(1, maximum/(double)Math.max(image.getWidth(), image.getHeight()));
        int width = Math.max(1, (int)Math.round(image.getWidth()*scale));
        int height = Math.max(1, (int)Math.round(image.getHeight()*scale));
        BufferedImage copy = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D graphics = copy.createGraphics();
        try {
            graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
            graphics.drawImage(image, 0, 0, width, height, null);
        } finally { graphics.dispose(); }
        return copy;
    }

    private static double[] readImage(File file) throws IOException {
        return prepareImage(file).pixels();
    }

    public static PreparedImage prepareImage(File file) throws IOException {
        if (!file.isFile() || file.length() == 0 || file.length() > 32_000_000) throw new IOException("invalid_image_file");
        try (InputStream input = new FileInputStream(file)) {
            int first = input.read(), second = input.read();
            if (first == 'P' && (second == '2' || second == '5')) return readPgm(file);
        }
        try (InputStream source = new FileInputStream(file);
             ImageInputStream input = new MemoryCacheImageInputStream(source)) {
            Iterator<ImageReader> readers = ImageIO.getImageReaders(input);
            if (!readers.hasNext()) throw new IOException("unsupported_image_format");
            ImageReader reader = readers.next();
            try {
                reader.setInput(input, true, true);
                int width = reader.getWidth(0), height = reader.getHeight(0);
                checkDimensions(width, height);
                BufferedImage image = reader.read(0);
                double[] result = new double[256];
                boolean gray = image.getColorModel().getColorSpace().getType() == ColorSpace.TYPE_GRAY;
                for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) {
                    // Toma el pixel del centro de cada celda de la cuadrícula 16x16.
                    int x = Math.min(width-1, (int)((c+.5)*width/16));
                    int y = Math.min(height-1, (int)((r+.5)*height/16));
                    double luminance, alpha = 255;
                    if (gray) {
                        // Lee el gris original para evitar cambios de color de getRGB.
                        int bits = image.getSampleModel().getSampleSize(0);
                        luminance = image.getRaster().getSampleDouble(x, y, 0)*255/(Math.pow(2, bits)-1);
                        if (image.getColorModel().hasAlpha()) {
                            int band = image.getRaster().getNumBands()-1;
                            int alphaBits = image.getSampleModel().getSampleSize(band);
                            alpha = image.getRaster().getSampleDouble(x, y, band)*255/(Math.pow(2, alphaBits)-1);
                        }
                    } else {
                        int argb = image.getRGB(x, y);
                        alpha = (argb >>> 24) & 255;
                        luminance = (299*((argb >>> 16)&255) + 587*((argb >>> 8)&255) + 114*(argb&255))/1000.0;
                    }
                    result[r*16+c] = luminance*alpha/255/255;
                }
                return new PreparedImage(file, width, height, result, image);
            } finally { reader.dispose(); }
        }
    }

    // Lee imágenes PGM, tanto de texto (P2) como binarias (P5).
    private static PreparedImage readPgm(File file) throws IOException {
        try (PushbackInputStream input = new PushbackInputStream(new BufferedInputStream(new FileInputStream(file)), 1)) {
            String magic = token(input);
            int width = integerToken(input), height = integerToken(input), maximum = integerToken(input);
            checkDimensions(width, height);
            if (maximum < 1 || maximum > 255) throw new IOException("unsupported_pgm_depth");
            double[] raw = new double[width*height];
            boolean binary = "P5".equals(magic);
            if (!binary && !"P2".equals(magic)) throw new IOException("invalid_pgm");
            if (binary) {
                // En P5, los datos empiezan justo después del espacio del encabezado.
                int separator = input.read();
                if (!whitespace(separator)) throw new IOException("invalid_pgm_separator");
                if (separator == '\r') {
                    int next = input.read();
                    if (next != '\n' && next != -1) input.unread(next);
                }
            }
            for (int i = 0; i < raw.length; i++) {
                int value = binary ? input.read() : integerToken(input);
                if (value < 0 || value > maximum) throw new IOException("invalid_pgm_samples");
                raw[i] = value*255.0/maximum;
            }
            if (binary ? input.read() != -1 : token(input) != null) throw new IOException("extra_pgm_samples");
            double[] result = new double[256];
            // Reduce la imagen original a los 256 valores que espera el modelo.
            for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++)
                result[r*16+c] = raw[Math.min(height-1, (int)((r+.5)*height/16))*width
                                    + Math.min(width-1, (int)((c+.5)*width/16))]/255;
            int pw = Math.min(480, width), ph = Math.max(1, (int)Math.round(height*(pw/(double)width)));
            if (ph > 480) { ph = 480; pw = Math.max(1, (int)Math.round(width*(480.0/height))); }
            BufferedImage preview = new BufferedImage(pw, ph, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < ph; y++) for (int x = 0; x < pw; x++) {
                int value = (int)Math.round(raw[Math.min(height-1, (int)((y+.5)*height/ph))*width
                                              +Math.min(width-1, (int)((x+.5)*width/pw))]);
                preview.setRGB(x, y, (value << 16) | (value << 8) | value);
            }
            return new PreparedImage(file, width, height, result, preview);
        }
    }

    private static void checkDimensions(int width, int height) throws IOException {
        if (width < 1 || height < 1 || (long)width*height > MAX_PIXELS) throw new IOException("invalid_image_dimensions");
    }

    private static boolean whitespace(int value) {
        return value == ' ' || value == '\r' || value == '\n' || value == '\t' || value == 11 || value == 12;
    }

    private static int integerToken(PushbackInputStream input) throws IOException {
        String value = token(input);
        if (value == null) throw new IOException("truncated_pgm");
        try { return Integer.parseInt(value); }
        catch (NumberFormatException ex) { throw new IOException("invalid_pgm_integer", ex); }
    }

    private static String token(PushbackInputStream input) throws IOException {
        int value;
        // Omite espacios y comentarios del encabezado PGM.
        while (true) {
            value = input.read();
            if (value == '#') { while (value != -1 && value != '\r' && value != '\n') value = input.read(); }
            else if (!whitespace(value)) break;
        }
        if (value == -1) return null;
        StringBuilder text = new StringBuilder();
        while (value != -1 && !whitespace(value) && value != '#') {
            if (text.length() >= 64) throw new IOException("invalid_pgm_token");
            text.append((char)value);
            value = input.read();
        }
        if (value != -1) input.unread(value);
        return text.toString();
    }
}
