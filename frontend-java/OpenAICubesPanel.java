import java.awt.*;
import java.io.*;
import java.net.ConnectException;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.filechooser.FileNameExtensionFilter;


public final class OpenAICubesPanel extends JPanel implements AutoCloseable {
    private static final int MAX_ACTIVE = 8;
    final JTextField host = new JTextField();
    final JTextField port = new JTextField();
    final JSpinner timeout = new JSpinner(new SpinnerNumberModel(15, 1, 120, 1));
    final JComboBox<ImageClient.Mode> mode = new JComboBox<>(ImageClient.Mode.values());
    final JButton choose = SwingUi.button("Seleccionar imagen…", false);
    final JButton send = SwingUi.button("Identificar imagen", true);
    final JButton clear = SwingUi.button("Limpiar", false);
    final JButton cancel = SwingUi.button("Cancelar pendientes", false);
    final JLabel connection = SwingUi.label("Sin comprobar", 12, true, new Color(199, 223, 232));
    final JLabel fileName = SwingUi.label("Ninguna imagen seleccionada", 12, true, SwingUi.INK);
    final JLabel imageInfo = SwingUi.label("PNG, JPEG o PGM", 11, false, SwingUi.MUTED);
    final JTextArea notice = SwingUi.text("Selecciona una imagen para identificarla.", 12, SwingUi.MUTED);
    final ImagePanel preview = new ImagePanel(260, 172);
    final ChatPanel chat = new ChatPanel();
    final ImageClient.Identity identity;
    final Map<String, QueryWorker> active = new LinkedHashMap<>();
    ClientCli.PreparedImage selected;
    boolean loading, closed;
    private int queryNumber;
    private SwingWorker<ClientCli.PreparedImage, Void> loader;

    public OpenAICubesPanel(String host, int port, ImageClient.Identity identity) {
        SwingUi.edt(); this.identity = identity;
        this.host.setText(host); this.port.setText(Integer.toString(port));
        this.host.getAccessibleContext().setAccessibleName("Host del servidor");
        this.port.getAccessibleContext().setAccessibleName("Puerto del servidor");
        timeout.getAccessibleContext().setAccessibleName("Tiempo límite en segundos");
        mode.getAccessibleContext().setAccessibleName("Modo de ejecución");
        setLayout(new BorderLayout()); setBackground(SwingUi.BACKGROUND);
        JPanel header = new JPanel(new BorderLayout(20, 0)); header.setBackground(SwingUi.INK);
        header.setBorder(new EmptyBorder(19, 24, 19, 24));
        JPanel brand = SwingUi.column();
        brand.add(SwingUi.label("OpenAICubes", 26, true, Color.WHITE));
        brand.add(SwingUi.label("Clasificación de imágenes", 12, false, new Color(185, 207, 222)));
        header.add(brand, BorderLayout.WEST);
        JPanel session = SwingUi.column();
        session.add(SwingUi.label("Cliente de demostración · "+identity.clientId, 11, false, new Color(201, 217, 231)));
        connection.setMaximumSize(new Dimension(Integer.MAX_VALUE, 22));
        session.add(Box.createVerticalStrut(7)); session.add(connection); header.add(session, BorderLayout.EAST);
        add(header, BorderLayout.NORTH);

        JPanel content = new JPanel(new BorderLayout(20, 0)); content.setOpaque(false);
        content.setBorder(new EmptyBorder(20, 22, 12, 22));
        JPanel sidebar = new SwingUi.Card(Color.WHITE, 18); sidebar.setLayout(new BorderLayout());
        JPanel controls = SwingUi.column(); controls.add(SwingUi.label("Nueva consulta", 18, true, SwingUi.INK));
        controls.add(Box.createVerticalStrut(14));
        JPanel endpoint = new JPanel(new GridLayout(1, 2, 10, 0)); endpoint.setOpaque(false);
        endpoint.add(field("Host del servidor", this.host)); endpoint.add(field("Puerto", this.port));
        controls.add(endpoint); controls.add(Box.createVerticalStrut(10));
        JPanel options = new JPanel(new GridLayout(1, 2, 10, 0)); options.setOpaque(false);
        options.add(field("Modo", mode)); options.add(field("Espera (segundos)", timeout));
        controls.add(options); controls.add(Box.createVerticalStrut(18));
        controls.add(preview); controls.add(Box.createVerticalStrut(10));
        fileName.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
        imageInfo.setMaximumSize(new Dimension(Integer.MAX_VALUE, 18));
        controls.add(fileName); controls.add(imageInfo); controls.add(Box.createVerticalStrut(12));
        choose.setAlignmentX(LEFT_ALIGNMENT); choose.setMaximumSize(new Dimension(Integer.MAX_VALUE, 42));
        send.setAlignmentX(LEFT_ALIGNMENT); send.setMaximumSize(new Dimension(Integer.MAX_VALUE, 44));
        controls.add(choose); controls.add(Box.createVerticalStrut(9)); controls.add(send);
        controls.add(Box.createVerticalStrut(12)); notice.setRows(3); controls.add(notice);
        // Alineacion uniforme en BoxLayout, incluido el preview y los grupos de campos.
        for (Component child : controls.getComponents()) if (child instanceof JComponent) ((JComponent)child).setAlignmentX(LEFT_ALIGNMENT);
        controls.setBorder(new EmptyBorder(0, 0, 8, 0)); sidebar.add(controls, BorderLayout.NORTH);
        JScrollPane sideScroll = new JScrollPane(sidebar); sideScroll.setBorder(null);
        sideScroll.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
        sideScroll.getVerticalScrollBar().setUnitIncrement(18); sideScroll.setPreferredSize(new Dimension(324, 600));
        content.add(sideScroll, BorderLayout.WEST);
        JPanel conversation = new JPanel(new BorderLayout(0, 12)); conversation.setOpaque(false);
        JPanel toolbar = new JPanel(new BorderLayout()); toolbar.setOpaque(false);
        toolbar.add(SwingUi.label("Conversación", 19, true, SwingUi.INK), BorderLayout.WEST);
        JPanel actions = new JPanel(new FlowLayout(FlowLayout.RIGHT, 7, 0)); actions.setOpaque(false);
        actions.add(cancel); actions.add(clear); toolbar.add(actions, BorderLayout.EAST);
        conversation.add(toolbar, BorderLayout.NORTH); conversation.add(chat, BorderLayout.CENTER);
        content.add(conversation, BorderLayout.CENTER); add(content, BorderLayout.CENTER);
        JLabel footer = SwingUi.label("Clases disponibles: Círculo · Cruz · Triángulo", 12, false, SwingUi.MUTED);
        footer.setBorder(new EmptyBorder(0, 24, 12, 24)); add(footer, BorderLayout.SOUTH);
        choose.addActionListener(event -> chooseImage()); send.addActionListener(event -> submit());
        clear.addActionListener(event -> { chat.clear(); notice.setText("Historial limpiado. Puedes identificar la imagen de nuevo."); });
        cancel.addActionListener(event -> {
            for (QueryWorker worker : new ArrayList<>(active.values())) { worker.client.close(); worker.cancel(true); }
        });
        updateButtons();
    }

    private static JPanel field(String name, JComponent component) {
        JPanel panel = new JPanel(new BorderLayout(0, 5)); panel.setOpaque(false);
        panel.add(SwingUi.label(name, 11, true, SwingUi.MUTED), BorderLayout.NORTH);
        component.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        component.setPreferredSize(new Dimension(110, 32)); panel.add(component, BorderLayout.CENTER);
        panel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 53)); return panel;
    }
    private void chooseImage() {
        SwingUi.edt();
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Seleccionar imagen para OpenAICubes"); chooser.setAcceptAllFileFilterUsed(false);
        chooser.setFileFilter(new FileNameExtensionFilter("Imágenes PNG, JPEG y PGM", "png", "jpg", "jpeg", "pgm"));
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) loadFile(chooser.getSelectedFile());
    }
    void loadFile(File file) {
        SwingUi.edt(); if (closed || loading) return;
        loading = true; selected = null; preview.setImage(null); fileName.setText("Cargando imagen…");
        imageInfo.setText("Preparando vista previa y entrada 16 × 16"); notice.setText("Leyendo la imagen…"); updateButtons();
        loader = new SwingWorker<ClientCli.PreparedImage, Void>() {
            @Override protected ClientCli.PreparedImage doInBackground() throws Exception { return ClientCli.prepareImage(file); }
            @Override protected void done() {
                if (closed) return;
                loading = false;
                try {
                    selected = get(); preview.setImage(selected.preview()); fileName.setText(selected.name);
                    fileName.setToolTipText(selected.name);
                    imageInfo.setText(selected.width+" × "+selected.height+" px · Entrada: 16 × 16 en gris");
                    notice.setForeground(SwingUi.MUTED); notice.setText("Imagen lista. Elige el modo y pulsa Identificar imagen.");
                } catch (Exception error) {
                    fileName.setText("No se pudo cargar la imagen"); imageInfo.setText("Selecciona otro archivo");
                    notice.setForeground(SwingUi.ERROR);
                    notice.setText("No se pudo leer la imagen. Usa un PNG, JPEG o PGM válido (hasta 32 MB y 4 megapíxeles).");
                }
                updateButtons();
            }
        };
        loader.execute();
    }

    void submit() {
        SwingUi.edt(); if (closed || loading || selected == null || active.size() >= MAX_ACTIVE) return;
        try {
            timeout.commitEdit();
            int serverPort = Integer.parseInt(port.getText().trim());
            ImageClient client = new ImageClient(host.getText().trim(), serverPort, ((Number)timeout.getValue()).intValue()*1000);
            ImageClient.Mode chosenMode = (ImageClient.Mode)mode.getSelectedItem();
            String id = identity.nextRequestId();
            ChatPanel.Entry entry = chat.addRequest(++queryNumber, id, selected, chosenMode, host.getText().trim()+":"+serverPort);
            QueryWorker worker = new QueryWorker(client, entry, selected.pixels(), chosenMode);
            active.put(id, worker); notice.setForeground(SwingUi.MUTED);
            notice.setText("Puedes enviar otra consulta o cambiar de imagen mientras se procesa.");
            connection.setText("Conectando…"); updateButtons(); worker.execute();
        } catch (Exception error) {
            notice.setForeground(SwingUi.ERROR); notice.setText("Revisa el host, el puerto (1–65535) y la espera (1–120 segundos).");
        }
    }
    private void updateButtons() {
        SwingUi.edt(); choose.setEnabled(!closed && !loading);
        send.setEnabled(!closed && !loading && selected != null && active.size() < MAX_ACTIVE);
        cancel.setEnabled(!closed && !active.isEmpty()); clear.setEnabled(!closed && active.isEmpty());
        send.setToolTipText(active.size() >= MAX_ACTIVE ? "Hay ocho consultas pendientes. Espera un resultado." : "Enviar la imagen al servidor");
    }
    final class QueryWorker extends SwingWorker<ImageClient.Response, ImageClient.Phase> {
        final ImageClient client;
        final ChatPanel.Entry entry;
        private final double[] pixels;
        private final ImageClient.Mode requestedMode;
        QueryWorker(ImageClient client, ChatPanel.Entry entry, double[] pixels, ImageClient.Mode mode) {
            this.client = client; this.entry = entry; this.pixels = pixels; this.requestedMode = mode;
        }
        @Override protected ImageClient.Response doInBackground() throws Exception {
            return client.identify(entry.requestId, pixels, requestedMode, phase -> publish(phase));
        }
        @Override protected void process(List<ImageClient.Phase> phases) {
            if (closed || isCancelled() || entry.finished || phases.isEmpty()) return;
            ImageClient.Phase phase = phases.get(phases.size()-1); entry.phase(phase);
            connection.setText(active.size() > 1 ? "Procesando · "+active.size()+" consultas" : entry.status.getText());
        }
        @Override protected void done() {
            client.close(); active.remove(entry.requestId);
            if (closed) return;
            try {
                ImageClient.Response response = get(); entry.complete(response);
                connection.setText("Resultado recibido");
            } catch (CancellationException error) {
                entry.fail("Consulta cancelada. Puedes volver a enviarla."); connection.setText("Consulta cancelada");
            } catch (Exception error) {
                Throwable cause = error instanceof ExecutionException ? error.getCause() : error;
                entry.fail(ImageClient.explain(cause));
                connection.setText(cause instanceof ClientCli.ServerError ? "El servidor respondió con un error" : "Error de conexión");
            }
            if (!active.isEmpty()) connection.setText("Procesando · "+active.size()+" consultas");
            chat.refresh(false); updateButtons();
        }
    }
    @Override public void close() {
        SwingUi.edt(); if (closed) return; closed = true;
        if (loader != null) loader.cancel(true);
        for (QueryWorker worker : new ArrayList<>(active.values())) {
            worker.client.close(); worker.cancel(true); worker.entry.fail("Consulta cancelada al cerrar la aplicación.");
        }
        active.clear(); updateButtons();
    }
}
