import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import javax.imageio.ImageIO;
import javax.swing.*;

/** Harness sin JUnit. Por defecto headless; --visual abre/cierra solo su JFrame. */
public final class FrontendChecks {
    private static void check(boolean value, String message) { if (!value) throw new AssertionError(message); }
    private interface Action { void run() throws Exception; }
    private static void rejects(Action action) throws Exception {
        try { action.run(); } catch (IOException | IllegalArgumentException expected) { return; }
        throw new AssertionError("La entrada invalida fue aceptada");
    }
    private static void edt(Action action) throws Exception {
        AtomicReference<Throwable> failed = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> { try { action.run(); } catch (Throwable error) { failed.set(error); } });
        if (failed.get() != null) throw new AssertionError("EDT", failed.get());
    }
    private interface Condition { boolean test(); }
    private static void await(Condition condition, int seconds) throws Exception {
        long deadline = System.nanoTime()+TimeUnit.SECONDS.toNanos(seconds);
        while (System.nanoTime() < deadline) {
            AtomicBoolean ready = new AtomicBoolean(); edt(() -> ready.set(condition.test()));
            if (ready.get()) return;
            Thread.sleep(20);
        }
        throw new AssertionError("Tiempo agotado esperando al componente Swing");
    }
    private static void layout(Container container) {
        container.doLayout();
        for (Component child : container.getComponents()) if (child instanceof Container) layout((Container)child);
    }
    private static void capture(OpenAICubesPanel panel, JFrame frame, Path path) throws Exception {
        Files.createDirectories(path.toAbsolutePath().getParent());
        if (frame != null) {
            edt(() -> { check(frame.isShowing() && frame.isDisplayable(), "JFrame no abierto"); frame.toFront(); });
            Thread.sleep(120);
            AtomicReference<Rectangle> bounds = new AtomicReference<>(); edt(() -> bounds.set(frame.getBounds()));
            ImageIO.write(new Robot().createScreenCapture(bounds.get()), "png", path.toFile());
        } else {
            edt(() -> {
                panel.setSize(1180, 830); layout(panel);
                BufferedImage image = new BufferedImage(1180, 830, BufferedImage.TYPE_INT_RGB);
                Graphics2D g = image.createGraphics();
                try { panel.printAll(g); } finally { g.dispose(); }
                ImageIO.write(image, "png", path.toFile());
            });
        }
    }
    private static JFileChooser findChooser(Container container) {
        if (container instanceof JFileChooser) return (JFileChooser)container;
        for (Component child : container.getComponents()) if (child instanceof Container) {
            JFileChooser found = findChooser((Container)child); if (found != null) return found;
        }
        return null;
    }
    private static JFileChooser visibleChooser() {
        for (Window window : Window.getWindows()) if (window.isShowing()) {
            JFileChooser chooser = findChooser(window); if (chooser != null) return chooser;
        }
        return null;
    }
    private static void unit(Path directory) throws Exception {
        Files.createDirectories(directory);
        ImageClient.Identity identity = new ImageClient.Identity();
        Set<String> ids = ConcurrentHashMap.newKeySet();
        ExecutorService threads = Executors.newFixedThreadPool(8);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < 8; t++) futures.add(threads.submit(() -> {
                for (int n = 0; n < 200; n++) {
                    String id = identity.nextRequestId();
                    check(!identity.clientId.equals(id) && ClientCli.validId(id), "ID invalido");
                    check(ids.add(id), "ID repetido");
                }
            }));
            for (Future<?> future : futures) future.get();
        } finally { threads.shutdownNow(); }
        check(ids.size() == 1600, "IDs concurrentes");
        double[] pixels = new double[256]; pixels[1] = .5; pixels[255] = 1;
        for (ImageClient.Mode mode : ImageClient.Mode.values()) {
            String message = ImageClient.message("image-1", pixels, mode);
            String[] fields = message.split("\\|", -1);
            check(fields.length == 4 && fields[0].equals("IDENTIFY") && fields[1].equals("image-1"), "Contrato IDENTIFY");
            check(fields[2].split(",").length == 256 && fields[3].equals(mode.name()), "Modo real en mensaje");
        }
        rejects(() -> ImageClient.message("bad id", pixels, ImageClient.Mode.SEQUENTIAL));
        rejects(() -> ImageClient.message("id", new double[255], ImageClient.Mode.SEQUENTIAL));
        rejects(() -> ImageClient.message("id", pixels, null));
        for (double value : new double[] {Double.NaN, Double.POSITIVE_INFINITY, -.1, 1.1}) {
            pixels[0] = value; rejects(() -> ImageClient.message("id", pixels, ImageClient.Mode.SEQUENTIAL));
        }
        rejects(() -> new ImageClient(" ", 9000, 10)); rejects(() -> new ImageClient("localhost", 0, 10));
        rejects(() -> new ImageClient("localhost", 9000, 0));
        ClientCli.Prediction prediction = ClientCli.parsePrediction("RESULT|id|1|3|0.2,0.3,0.5", "id");
        check(prediction.index == 2 && prediction.confidence() == .5 && prediction.label().equals("triangulo"), "Parseo");
        prediction.probabilities()[0] = 99; check(prediction.probabilities()[0] == .2, "Probabilidades inmutables");
        for (String reply : Arrays.asList("RESULT|other|1|3|0.2,0.3,0.5", "RESULT|id|3|1|0.2,0.3,0.5",
                "RESULT|id|1|3|0.2,0.3", "RESULT|id|1|3|0.2,0.3,0.4", "RESULT|id|1|3|NaN,0,1",
                "RESULT|id|1|3|-1,1,1", "RESULT|id|1|3|1f,0,0", "ERROR|other|offline", "garbage"))
            rejects(() -> ClientCli.parsePrediction(reply, "id"));
        try { ClientCli.parsePrediction("ERROR|id|insufficient_workers", "id"); throw new AssertionError("ERROR ignorado"); }
        catch (ClientCli.ServerError expected) { check(ImageClient.explain(expected).contains("dos workers"), "Error comprensible"); }
        BufferedImage original = new BufferedImage(31, 19, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 19; y++) for (int x = 0; x < 31; x++) original.setRGB(x, y, (x*8 << 16) | (y*13 << 8) | ((x+y)*5));
        File jpeg = directory.resolve("standard.jpg").toFile();
        check(ImageIO.write(original, "jpg", jpeg), "Escritor JPEG estandar");
        BufferedImage decoded = ImageIO.read(jpeg);
        ClientCli.PreparedImage prepared = ClientCli.prepareImage(jpeg);
        check(prepared.width == 31 && prepared.height == 19 && prepared.pixels().length == 256, "Dimensiones JPEG");
        double[] actual = prepared.pixels();
        for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) {
            int rgb = decoded.getRGB((int)((c+.5)*31/16), (int)((r+.5)*19/16));
            double expected = (299*((rgb >>> 16)&255)+587*((rgb >>> 8)&255)+114*(rgb&255))/255000.0;
            check(Math.abs(expected-actual[r*16+c]) < 1e-12, "Conversion JPEG");
        }
        prepared.pixels()[0] = 99; check(prepared.pixels()[0] <= 1, "Snapshot pixeles");
        BufferedImage preview = prepared.preview(); int rgb = preview.getRGB(0, 0); preview.setRGB(0, 0, ~rgb);
        check(prepared.preview().getRGB(0, 0) == rgb, "Snapshot preview");
        Path invalid = directory.resolve("fake.png"); Files.write(invalid, new byte[] {1, 2, 3});
        rejects(() -> ClientCli.prepareImage(invalid.toFile())); rejects(() -> ClientCli.prepareImage(directory.toFile()));
        try { new ImagePanel(100, 100).setImage(decoded); throw new AssertionError("Actualizacion fuera del EDT aceptada"); }
        catch (IllegalStateException expected) { }
        AtomicReference<OpenAICubesPanel> view = new AtomicReference<>();
        edt(() -> view.set(new OpenAICubesPanel("127.0.0.1", 9000, identity)));
        OpenAICubesPanel panel = view.get();
        try {
            capture(panel, null, directory.resolve("frontend-empty.png"));
            edt(() -> { check(!panel.send.isEnabled(), "Enviar sin imagen"); panel.loadFile(jpeg); });
            await(() -> !panel.loading, 10);
            edt(() -> { check(panel.selected != null && panel.send.isEnabled(), "Carga JPEG asincrona"); });
            capture(panel, null, directory.resolve("frontend-ready.png"));
            edt(() -> {
                ChatPanel.Entry pending = panel.chat.addRequest(1, "pending", panel.selected, ImageClient.Mode.SEQUENTIAL, "localhost");
                for (int i = 2; i <= 101; i++) {
                    ChatPanel.Entry entry = panel.chat.addRequest(i, "history-"+i, panel.selected, ImageClient.Mode.SEQUENTIAL, "localhost");
                    entry.fail("Entrada terminada de prueba.");
                }
                check(panel.chat.entries.size() == 100 && panel.chat.entries.contains(pending), "Historial elimino consulta activa");
                pending.fail("Consulta de prueba cancelada."); panel.chat.clear();
            });
            edt(() -> panel.loadFile(invalid.toFile())); await(() -> !panel.loading, 10);
            edt(() -> check(panel.selected == null && !panel.send.isEnabled() && panel.notice.getText().contains("No se pudo leer"), "Imagen invalida"));
        } finally { edt(panel::close); }
        System.out.println("UNIT_OK|1600_unique_ids|JPEG|validation|message|parser|EDT|async_load");
    }

    private static void flow(String host, int port, Path examples, Path directory, boolean visual) throws Exception {
        ImageClient.Identity identity = new ImageClient.Identity();
        AtomicReference<OpenAICubesPanel> view = new AtomicReference<>(); AtomicReference<JFrame> window = new AtomicReference<>();
        AtomicInteger ticks = new AtomicInteger(); AtomicReference<javax.swing.Timer> pulse = new AtomicReference<>();
        edt(() -> {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            OpenAICubesPanel panel;
            if (visual) {
                JFrame frame = OpenAICubesGUI.open(host, port, identity);
                panel = (OpenAICubesPanel)frame.getContentPane(); window.set(frame);
            } else panel = new OpenAICubesPanel(host, port, identity);
            view.set(panel);
            javax.swing.Timer timer = new javax.swing.Timer(20, event -> ticks.incrementAndGet()); timer.start(); pulse.set(timer);
        });
        OpenAICubesPanel panel = view.get();
        try {
            capture(panel, window.get(), directory.resolve("frontend-start.png"));
            for (String name : new String[] {"circulo", "cruz", "triangulo"}) {
                File file = examples.resolve(name+"-0.png").toFile();
                if (visual && name.equals("circulo")) {
                    SwingUtilities.invokeLater(() -> panel.choose.doClick(0));
                    await(() -> visibleChooser() != null, 10);
                    edt(() -> { JFileChooser chooser = visibleChooser(); chooser.setSelectedFile(file); chooser.approveSelection(); });
                    System.out.println("FILE_CHOOSER_OK|"+file.getName());
                } else edt(() -> panel.loadFile(file));
                await(() -> !panel.loading && panel.selected != null, 10);
                for (ImageClient.Mode mode : ImageClient.Mode.values()) edt(() -> {
                    check(panel.selected != null, "Imagen valida"); panel.mode.setSelectedItem(mode); panel.send.doClick(0);
                });
            }
            edt(() -> check(panel.chat.entries.size() == 6, "Seis consultas aceptadas"));
            capture(panel, window.get(), directory.resolve("frontend-processing.png"));
            await(() -> panel.active.isEmpty(), 30);
            edt(() -> {
                check(ticks.get() >= 3, "EDT no atendio eventos durante la red");
                Set<String> ids = new HashSet<>();
                for (ChatPanel.Entry entry : panel.chat.entries) {
                    check(entry.finished && entry.response != null && entry.error == null, "Consulta fallo: "+entry.error);
                    check(!entry.busy.isIndeterminate(), "Animacion pendiente");
                    check(ids.add(entry.requestId) && !entry.requestId.equals(identity.clientId), "IDs aislados");
                    String expected = entry.fileName.substring(0, entry.fileName.indexOf('-'));
                    check(entry.response.prediction.label().equals(expected), "Se mezclaron respuestas o imagenes");
                    System.out.println("FLOW|"+entry.fileName+"|"+entry.mode+"|"+entry.response.prediction.reply);
                }
                check(panel.send.isEnabled() && panel.clear.isEnabled() && !panel.cancel.isEnabled(), "Controles al terminar");
            });
            capture(panel, window.get(), directory.resolve("frontend-results.png"));
            System.out.println("FLOW_OK|6|EDT_ticks="+ticks.get()+"|visual="+visual);
        } finally { edt(() -> { pulse.get().stop(); panel.close(); if (window.get() != null) window.get().dispose(); }); }
    }

    private static void error(String host, int port, Path image, int timeout, boolean cancel) throws Exception {
        ImageClient.Identity identity = new ImageClient.Identity(); AtomicReference<OpenAICubesPanel> view = new AtomicReference<>();
        edt(() -> { OpenAICubesPanel panel = new OpenAICubesPanel(host, port, identity); panel.timeout.setValue(timeout); view.set(panel); panel.loadFile(image.toFile()); });
        OpenAICubesPanel panel = view.get();
        try {
            await(() -> !panel.loading, 10); long start = System.nanoTime();
            edt(() -> {
                panel.send.doClick(0);
                if (cancel) {
                    for (int i = 0; i < 8; i++) panel.send.doClick(0);
                    check(panel.active.size() == 8 && panel.chat.entries.size() == 8 && !panel.send.isEnabled(), "Limite de consultas pendientes");
                }
            });
            if (cancel) { Thread.sleep(150); edt(() -> panel.cancel.doClick(0)); }
            await(() -> panel.active.isEmpty(), timeout+4);
            edt(() -> {
                for (ChatPanel.Entry entry : panel.chat.entries) check(entry.finished && entry.error != null && !entry.busy.isIndeterminate(), "Error sin mostrar");
                check(panel.send.isEnabled(), "No se puede reintentar");
                System.out.println("UI_ERROR|"+panel.chat.entries.get(0).error+"|elapsedMs="+(System.nanoTime()-start)/1_000_000.0);
            });
        } finally { edt(panel::close); }
    }
    public static void main(String[] args) {
        try {
            // Windows puede elegir una codificacion de consola distinta de UTF-8.
            System.setOut(new java.io.PrintStream(System.out, true, "UTF-8"));
            System.setErr(new java.io.PrintStream(System.err, true, "UTF-8"));
            switch (args[0]) {
                case "unit": unit(Paths.get(args[1])); break;
                case "image": {
                    double[] pixels = ClientCli.prepareImage(new File(args[1])).pixels();
                    StringBuilder text = new StringBuilder("PIXELS|");
                    for (double value : pixels) { if (text.length() > 7) text.append(','); text.append(value); }
                    System.out.println(text); break;
                }
                case "request": {
                    ImageClient.Identity identity = new ImageClient.Identity();
                    try (ImageClient client = new ImageClient(args[1], Integer.parseInt(args[2]), Integer.parseInt(args[5]))) {
                        ImageClient.Response response = client.identify(identity.nextRequestId(), ClientCli.prepareImage(new File(args[4])).pixels(), ImageClient.Mode.valueOf(args[3]), phase -> System.out.println("PHASE|"+phase));
                        System.out.println(response.prediction.reply); System.out.println("CLASS|"+response.prediction.label());
                    }
                    break;
                }
                case "flow": flow(args[1], Integer.parseInt(args[2]), Paths.get(args[3]), Paths.get(args[4]), args.length == 6 && args[5].equals("--visual")); break;
                case "error": error(args[1], Integer.parseInt(args[2]), Paths.get(args[3]), Integer.parseInt(args[4]), args.length == 6 && args[5].equals("cancel")); break;
                default: throw new IllegalArgumentException("unknown test command");
            }
        } catch (Throwable error) { error.printStackTrace(); System.exit(1); }
    }
}
