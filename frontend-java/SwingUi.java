import java.awt.*;
import javax.swing.*;
import javax.swing.border.EmptyBorder;


final class SwingUi {
    static final Color INK = new Color(24, 39, 62), MUTED = new Color(96, 113, 135),
        ACCENT = new Color(0, 120, 117), BACKGROUND = new Color(241, 245, 250),
        LINE = new Color(220, 228, 237), ERROR = new Color(174, 47, 59);
    static void edt() {
        if (!SwingUtilities.isEventDispatchThread()) throw new IllegalStateException("Swing must be updated on the EDT");
    }
    static JLabel label(String text, int size, boolean bold, Color color) {
        JLabel label = new JLabel(text);
        label.putClientProperty("html.disable", Boolean.TRUE);
        label.setFont(new Font("Segoe UI", bold ? Font.BOLD : Font.PLAIN, size));
        label.setForeground(color);
        return label;
    }
    static JTextArea text(String text, int size, Color color) {
        JTextArea area = new JTextArea(text);
        area.setEditable(false); area.setLineWrap(true); area.setWrapStyleWord(true);
        area.setFont(new Font("Segoe UI", Font.PLAIN, size)); area.setForeground(color);
        area.setOpaque(false); area.setFocusable(true);
        return area;
    }
    static JPanel column() {
        JPanel panel = new JPanel(); panel.setOpaque(false);
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS)); return panel;
    }
    static final class Card extends JPanel {
        private final Color fill;
        Card(Color fill, int padding) {
            this.fill = fill; setOpaque(false); setBorder(new EmptyBorder(padding, padding, padding, padding));
        }
        @Override protected void paintComponent(Graphics graphics) {
            Graphics2D g = (Graphics2D)graphics.create();
            try {
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g.setColor(fill); g.fillRoundRect(0, 0, getWidth()-1, getHeight()-1, 18, 18);
                g.setColor(LINE); g.drawRoundRect(0, 0, getWidth()-1, getHeight()-1, 18, 18);
            } finally { g.dispose(); }
            super.paintComponent(graphics);
        }
    }
    static JButton button(String text, boolean primary) {
        JButton button = new JButton(text) {
            @Override protected void paintComponent(Graphics graphics) {
                Graphics2D g = (Graphics2D)graphics.create();
                try {
                    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                    g.setColor(!isEnabled() ? LINE : primary ? ACCENT : Color.WHITE);
                    g.fillRoundRect(0, 0, getWidth(), getHeight(), 12, 12);
                    if (!primary || isFocusOwner()) { g.setColor(isFocusOwner() ? ACCENT.brighter() : LINE); g.drawRoundRect(1, 1, getWidth()-3, getHeight()-3, 12, 12); }
                } finally { g.dispose(); }
                super.paintComponent(graphics);
            }
        };
        button.setContentAreaFilled(false); button.setBorderPainted(false); button.setFocusPainted(false);
        button.setOpaque(false); button.setFont(new Font("Segoe UI", Font.BOLD, 13));
        button.setForeground(primary ? Color.WHITE : INK); button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        button.setBorder(new EmptyBorder(11, 15, 11, 15));
        return button;
    }
}
