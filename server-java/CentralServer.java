import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/** Coordinador de prueba: múltiples clientes y delegación round-robin de un micro chunk completo. */
public class CentralServer {
    private static final int MAX_BYTES = 2_000_000;
    private static final AtomicInteger cursor = new AtomicInteger();
    private static final List<InetSocketAddress> workers = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Uso: java CentralServer <puerto> <hostWorker:puerto> [otros...]");
            return;
        }
        int port = Integer.parseInt(args[0]);
        for (int i = 1; i < args.length; i++) {
            int idx = args[i].lastIndexOf(':');
            if (idx < 1) throw new IllegalArgumentException("worker debe ser host:puerto");
            workers.add(new InetSocketAddress(args[i].substring(0,idx), Integer.parseInt(args[i].substring(idx+1))));
        }
        ExecutorService clients = Executors.newFixedThreadPool(16);
        try (ServerSocket listener = new ServerSocket(port)) {
            System.out.println("CentralServer en " + port + ", workers=" + workers);
            while (true) {
                Socket socket = listener.accept();
                clients.execute(() -> handle(socket));
            }
        } finally { clients.shutdown(); }
    }

    private static void handle(Socket socket) {
        try (Socket client = socket;
             DataInputStream input = new DataInputStream(new BufferedInputStream(client.getInputStream()));
             DataOutputStream output = new DataOutputStream(new BufferedOutputStream(client.getOutputStream()))) {
            client.setSoTimeout(30000);
            while (true) {
                String request;
                try { request = read(input); }
                catch (EOFException ex) { return; }
                String[] fields = request.split("\\|", -1);
                String id = fields.length > 1 ? fields[1] : "unknown";
                String reply;
                if (fields.length >= 2 && Arrays.asList("PING", "CONV", "MLP").contains(fields[0])) {
                    try { reply = forward(request); }
                    catch (IOException ex) { reply = "ERROR|" + id + "|no_worker_available"; }
                } else reply = "ERROR|" + id + "|unsupported_command";
                write(output, reply);
            }
        } catch (IOException ex) { System.err.println("Cliente desconectado: " + ex.getMessage()); }
    }

    private static String forward(String request) throws IOException {
        int first = Math.floorMod(cursor.getAndIncrement(), workers.size());
        IOException last = new IOException("no_worker_available");
        for (int k = 0; k < workers.size(); k++) {
            InetSocketAddress endpoint = workers.get((first + k) % workers.size());
            try (Socket worker = new Socket()) {
                worker.connect(endpoint, 2000);
                worker.setSoTimeout(10000);
                DataOutputStream out = new DataOutputStream(new BufferedOutputStream(worker.getOutputStream()));
                DataInputStream in = new DataInputStream(new BufferedInputStream(worker.getInputStream()));
                write(out, request);
                return read(in);
            } catch (IOException ex) {
                last = ex;
                System.err.println("Worker falló: " + endpoint + " : " + ex.getMessage());
            }
        }
        throw last;
    }

    private static String read(DataInputStream in) throws IOException {
        int len = in.readInt();
        if (len <= 0 || len > MAX_BYTES) throw new IOException("invalid_length");
        byte[] data = new byte[len];
        in.readFully(data);
        return new String(data, StandardCharsets.UTF_8);
    }
    private static void write(DataOutputStream out, String s) throws IOException {
        byte[] data = s.getBytes(StandardCharsets.UTF_8);
        if (data.length > MAX_BYTES) throw new IOException("too_large");
        out.writeInt(data.length);
        out.write(data);
        out.flush();
    }
}
