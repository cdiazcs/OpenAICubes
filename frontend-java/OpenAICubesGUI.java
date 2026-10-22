import java.awt.*;
import java.awt.event.*;
import javax.swing.*;


public final class OpenAICubesGUI {
    public static void main(String[] args) {
        try {
            if (args.length > 2) throw new IllegalArgumentException("Uso: OpenAICubesGUI [host puerto]");
            String host = args.length > 0 ? args[0] : "127.0.0.1";
            int port = args.length > 1 ? Integer.parseInt(args[1]) : 9000;
            new ImageClient(host, port, 15000).close(); // Validacion local; no abre sockets.
            if (GraphicsEnvironment.isHeadless()) throw new IllegalStateException("No hay una pantalla gráfica disponible.");
            // UUID puede inicializar la entropia lentamente en Windows: nunca hacerlo en el EDT.
            ImageClient.Identity identity = new ImageClient.Identity();
            SwingUtilities.invokeLater(() -> open(host, port, identity));
        } catch (Exception error) {
            System.err.println("ERROR: "+error.getMessage()); System.exit(1);
        }
    }

    static JFrame open(String host, int port, ImageClient.Identity identity) {
        SwingUi.edt();
        try { UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); }
        catch (Exception ignored) { }
        JFrame frame = new JFrame("OpenAICubes");
        OpenAICubesPanel panel = new OpenAICubesPanel(host, port, identity);
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        frame.setContentPane(panel); frame.setMinimumSize(new Dimension(1000, 740)); frame.setSize(1180, 830);
        frame.setLocationRelativeTo(null);
        frame.addWindowListener(new WindowAdapter() {
            @Override public void windowClosing(WindowEvent event) { panel.close(); }
            @Override public void windowClosed(WindowEvent event) { panel.close(); }
        });
        frame.setVisible(true);
        return frame;
    }
}
