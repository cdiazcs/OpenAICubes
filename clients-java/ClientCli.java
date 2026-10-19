import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;

/** Cliente de prueba, NO es el frontend Swing solicitado para la entrega. */
public class ClientCli {
    public static void main(String[] args) throws Exception {
        String host = args.length > 0 ? args[0] : "127.0.0.1";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 9000;
        String[] cases = {
            "PING|demo0",
            "CONV|demo1|3|3|2|2|1,2,3,4,5,6,7,8,9|1,0,0,-1",
            "MLP|demo2|2|2|2|1,2|1,0,0,1|0,0|1,0,0,1|0,0"
        };
        try (Socket socket = new Socket(host,port);
             DataOutputStream out = new DataOutputStream(socket.getOutputStream());
             DataInputStream in = new DataInputStream(socket.getInputStream())) {
            socket.setSoTimeout(15000);
            for (String s : cases) {
                byte[] req = s.getBytes(StandardCharsets.UTF_8);
                out.writeInt(req.length);
                out.write(req);
                out.flush();
                int len = in.readInt();
                if (len <= 0 || len > 2_000_000) throw new IOException("invalid_response_length");
                byte[] response = new byte[len];
                in.readFully(response);
                System.out.println(s + "\n  => " + new String(response, StandardCharsets.UTF_8) + "\n");
            }
        }
    }
}
