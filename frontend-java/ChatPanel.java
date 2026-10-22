import java.awt.*;
import java.util.*;
import java.util.List;
import javax.swing.*;
import javax.swing.border.EmptyBorder;

/** Entradas independientes: la llegada de una respuesta nunca elige otra burbuja. */
public final class ChatPanel extends JPanel {
    final List<Entry> entries = new ArrayList<>();
    private final JPanel history = SwingUi.column();
    private final JScrollPane scroll = new JScrollPane(history);
    private final JLabel welcome = SwingUi.label("Selecciona una imagen para comenzar", 19, true, SwingUi.INK);

    public ChatPanel() {
        SwingUi.edt(); setLayout(new BorderLayout()); setOpaque(false);
        history.setBorder(new EmptyBorder(4, 4, 16, 12));
        JPanel greeting = new SwingUi.Card(Color.WHITE, 24);
        greeting.setLayout(new BorderLayout(0, 12)); greeting.add(welcome, BorderLayout.NORTH);
        greeting.add(SwingUi.text("Identifica círculos, cruces y triángulos. Puedes consultar la misma imagen "
            +"con ambos modos; cada respuesta aparecerá junto a su imagen.", 14, SwingUi.MUTED), BorderLayout.CENTER);
        greeting.setMaximumSize(new Dimension(Integer.MAX_VALUE, 150));
        history.add(greeting);
        scroll.setBorder(null); scroll.setOpaque(false); scroll.getViewport().setOpaque(false);
        scroll.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
        scroll.getVerticalScrollBar().setUnitIncrement(22);
        getAccessibleContext().setAccessibleName("Historial de consultas");
        add(scroll, BorderLayout.CENTER);
    }
    Entry addRequest(int number, String id, ClientCli.PreparedImage image, ImageClient.Mode mode, String endpoint) {
        SwingUi.edt();
        if (entries.isEmpty()) history.removeAll();
        // No se elimina ninguna consulta activa; el panel limita a ocho las pendientes.
        while (entries.size() >= 100) {
            Entry oldestFinished = null;
            for (Entry previous : entries) if (previous.finished) { oldestFinished = previous; break; }
            if (oldestFinished == null) throw new IllegalStateException("history_full");
            entries.remove(oldestFinished); history.remove(oldestFinished);
        }
        Entry entry = new Entry(number, id, image, mode, endpoint);
        entries.add(entry); history.add(entry); refresh(true); return entry;
    }
    void clear() {
        SwingUi.edt(); entries.clear(); history.removeAll();
        history.add(SwingUi.label("Historial vacío. Selecciona una imagen o vuelve a identificarla.", 15, false, SwingUi.MUTED));
        refresh(false);
    }
    void refresh(boolean forceBottom) {
        SwingUi.edt();
        JScrollBar bar = scroll.getVerticalScrollBar();
        boolean atBottom = bar.getValue()+bar.getVisibleAmount() >= bar.getMaximum()-70;
        history.revalidate(); history.repaint();
        if (forceBottom || atBottom) SwingUtilities.invokeLater(() -> bar.setValue(bar.getMaximum()));
    }

    static final class Entry extends JPanel {
        final String requestId;
        final ImageClient.Mode mode;
        final String fileName;
        final JLabel status = SwingUi.label("Conectando…", 13, false, SwingUi.MUTED);
        final JPanel answer = SwingUi.column();
        final JProgressBar busy = new JProgressBar();
        ImageClient.Response response;
        String error;
        boolean finished;

        Entry(int number, String id, ClientCli.PreparedImage image, ImageClient.Mode mode, String endpoint) {
            this.requestId = id; this.mode = mode; this.fileName = image.name;
            setOpaque(false); setLayout(new BorderLayout(0, 12));
            setBorder(new EmptyBorder(10, 0, 18, 0));
            JPanel user = new SwingUi.Card(new Color(224, 240, 247), 14);
            user.setLayout(new BorderLayout(14, 0));
            ImagePanel preview = new ImagePanel(88, 88);
            preview.setImage(ClientCli.thumbnail(image.preview(), 112)); user.add(preview, BorderLayout.WEST);
            JPanel details = SwingUi.column();
            details.add(SwingUi.label("TÚ  ·  Consulta "+String.format(Locale.ROOT, "%03d", number), 11, true, SwingUi.MUTED));
            details.add(Box.createVerticalStrut(5));
            details.add(SwingUi.label("Identificar imagen", 16, true, SwingUi.INK));
            JLabel name = SwingUi.label(image.name, 12, false, SwingUi.INK); name.setToolTipText(image.name);
            details.add(name); details.add(SwingUi.label(mode.name(), 11, true, SwingUi.ACCENT));
            details.setToolTipText("Solicitud: "+id+" · Servidor: "+endpoint);
            user.add(details, BorderLayout.CENTER);
            JPanel userRow = new JPanel(new BorderLayout()); userRow.setOpaque(false);
            userRow.setBorder(new EmptyBorder(0, 75, 0, 0)); userRow.add(user, BorderLayout.CENTER);
            add(userRow, BorderLayout.NORTH);

            JPanel bot = new SwingUi.Card(Color.WHITE, 18); bot.setLayout(new BorderLayout(0, 9));
            bot.add(SwingUi.label("OPENAICUBES", 11, true, SwingUi.ACCENT), BorderLayout.NORTH);
            answer.add(status); busy.setIndeterminate(true); busy.setPreferredSize(new Dimension(120, 4));
            busy.setMaximumSize(new Dimension(Integer.MAX_VALUE, 4));
            answer.add(Box.createVerticalStrut(8)); answer.add(busy);
            bot.add(answer, BorderLayout.CENTER);
            JPanel botRow = new JPanel(new BorderLayout()); botRow.setOpaque(false);
            botRow.setBorder(new EmptyBorder(0, 0, 0, 35)); botRow.add(bot, BorderLayout.CENTER);
            add(botRow, BorderLayout.CENTER);
            getAccessibleContext().setAccessibleName("Consulta "+number+" "+mode);
        }
        void phase(ImageClient.Phase phase) {
            SwingUi.edt(); if (finished) return;
            switch (phase) {
                case CONNECTING: status.setText("Conectando…"); break;
                case SENDING: status.setText("Enviando imagen…"); break;
                case PROCESSING: status.setText("Procesando · "+mode.name()+"…"); break;
                case RECEIVED: status.setText("Resultado recibido"); break;
            }
        }
        void complete(ImageClient.Response response) {
            SwingUi.edt(); if (finished) return;
            if (!requestId.equals(response.requestId) || mode != response.mode) throw new IllegalArgumentException("response_mismatch");
            this.response = response; finished = true; busy.setIndeterminate(false); answer.removeAll();
            String[] names = {"Círculo", "Cruz", "Triángulo"};
            answer.add(SwingUi.label(names[response.prediction.index], 26, true, SwingUi.INK));
            answer.add(SwingUi.label(String.format(Locale.ROOT, "Confianza: %.2f %%  ·  Tiempo: %.3f ms",
                response.prediction.confidence()*100, response.millis), 12, false, SwingUi.MUTED));
            answer.add(Box.createVerticalStrut(10));
            double[] values = response.prediction.probabilities();
            for (int i = 0; i < 3; i++) {
                JPanel row = new JPanel(new BorderLayout(10, 0)); row.setOpaque(false);
                JLabel label = SwingUi.label(names[i], 12, i == response.prediction.index, SwingUi.INK);
                label.setPreferredSize(new Dimension(67, 20)); row.add(label, BorderLayout.WEST);
                JProgressBar bar = new JProgressBar(0, 10000); bar.setValue((int)Math.round(values[i]*10000));
                bar.setStringPainted(true); bar.setString(String.format(Locale.ROOT, "%.4f %%", values[i]*100));
                bar.setForeground(i == response.prediction.index ? SwingUi.ACCENT : new Color(145, 178, 195));
                bar.setBackground(SwingUi.BACKGROUND); bar.setBorderPainted(false);
                bar.getAccessibleContext().setAccessibleName("Probabilidad de "+names[i]);
                bar.setToolTipText("Probabilidad: "+Double.toString(values[i]));
                row.add(bar, BorderLayout.CENTER); row.setMaximumSize(new Dimension(Integer.MAX_VALUE, 23));
                answer.add(row); answer.add(Box.createVerticalStrut(4));
            }
            answer.add(SwingUi.label("Resultado recibido · "+mode.name(), 11, false, SwingUi.ACCENT));
            for (Component child : answer.getComponents()) if (child instanceof JComponent) ((JComponent)child).setAlignmentX(LEFT_ALIGNMENT);
            revalidate(); repaint();
        }
        void fail(String message) {
            SwingUi.edt(); if (finished) return;
            finished = true; error = message; busy.setIndeterminate(false); answer.removeAll();
            answer.add(SwingUi.label("No se pudo identificar", 17, true, SwingUi.ERROR));
            JTextArea detail = SwingUi.text(message, 13, SwingUi.MUTED); detail.setRows(2);
            answer.add(detail); revalidate(); repaint();
        }
        @Override public Dimension getMaximumSize() { return new Dimension(Integer.MAX_VALUE, getPreferredSize().height); }
    }
}
