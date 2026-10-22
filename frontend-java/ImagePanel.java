import java.awt.*;
import java.awt.image.BufferedImage;
import javax.swing.*;


public final class ImagePanel extends JPanel {
    private BufferedImage image;
    public ImagePanel(int width, int height) {
        setPreferredSize(new Dimension(width, height));
        setMinimumSize(new Dimension(80, 80));
        setOpaque(false);
        getAccessibleContext().setAccessibleName("Vista previa de la imagen");
    }
    public void setImage(BufferedImage image) {
        SwingUi.edt();
        this.image = image;
        repaint();
    }
    @Override protected void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);
        Graphics2D g = (Graphics2D)graphics.create();
        try {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(new Color(23, 34, 52));
            g.fillRoundRect(0, 0, getWidth(), getHeight(), 20, 20);
            if (image == null) {
                int x = getWidth()/2, y = getHeight()/2;
                g.setStroke(new BasicStroke(2)); g.setColor(new Color(122, 145, 171));
                g.drawRoundRect(x-23, y-26, 46, 40, 7, 7);
                g.drawOval(x-13, y-18, 8, 8);
                g.drawPolyline(new int[] {x-18, x-4, x+5, x+18}, new int[] {y+8, y-5, y+3, y-13}, 4);
            } else {
                double scale = Math.min((getWidth()-20)/(double)image.getWidth(), (getHeight()-20)/(double)image.getHeight());
                int w = Math.max(1, (int)(image.getWidth()*scale)), h = Math.max(1, (int)(image.getHeight()*scale));
                g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
                g.drawImage(image, (getWidth()-w)/2, (getHeight()-h)/2, w, h, null);
            }
        } finally { g.dispose(); }
    }
}
