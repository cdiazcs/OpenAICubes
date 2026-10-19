// Worker C# (.NET 8): sin NuGet ni librerías externas.
using System;
using System.Buffers.Binary;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

static class Worker
{
    private const int MaxBytes = 2_000_000;
    static async Task Main(string[] args)
    {
        int port = args.Length > 0 ? int.Parse(args[0], CultureInfo.InvariantCulture) : 9103;
        var listener = new TcpListener(IPAddress.Any, port);
        listener.Start();
        Console.WriteLine($"WorkerCSharp escuchando en {port}");
        while (true)
        {
            TcpClient client = await listener.AcceptTcpClientAsync();
            _ = Task.Run(() => Handle(client));
        }
    }

    static void Handle(TcpClient client)
    {
        using (client)
        {
            client.ReceiveTimeout = 15000;
            client.SendTimeout = 15000;
            using NetworkStream stream = client.GetStream();
            try
            {
                while (true)
                {
                    string? request = ReadFrame(stream);
                    if (request is null) return;
                    string response;
                    try { response = Evaluate(request); }
                    catch (Exception ex) when (ex is FormatException || ex is ArgumentException || ex is OverflowException)
                    {
                        string[] parts = request.Split('|');
                        string id = parts.Length > 1 ? parts[1] : "unknown";
                        response = $"ERROR|{id}|{ex.GetType().Name}";
                    }
                    WriteFrame(stream, response);
                }
            }
            catch (IOException ex) { Console.Error.WriteLine($"conexion cerrada: {ex.Message}"); }
        }
    }

    static string? ReadFrame(Stream stream)
    {
        byte[] header = new byte[4];
        if (!ReadExact(stream, header, allowEof: true)) return null;
        uint length = BinaryPrimitives.ReadUInt32BigEndian(header);
        if (length == 0 || length > MaxBytes) throw new IOException("invalid_length");
        byte[] data = new byte[(int)length];
        ReadExact(stream, data, allowEof: false);
        return Encoding.UTF8.GetString(data);
    }
    static bool ReadExact(Stream stream, byte[] data, bool allowEof)
    {
        int read = 0;
        while (read < data.Length)
        {
            int n = stream.Read(data, read, data.Length - read);
            if (n == 0)
            {
                if (allowEof && read == 0) return false;
                throw new IOException("truncated_frame");
            }
            read += n;
        }
        return true;
    }
    static void WriteFrame(Stream stream, string text)
    {
        byte[] payload = Encoding.UTF8.GetBytes(text);
        if (payload.Length > MaxBytes) throw new IOException("response_too_large");
        byte[] header = new byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(header, (uint)payload.Length);
        stream.Write(header, 0, header.Length);
        stream.Write(payload, 0, payload.Length);
        stream.Flush();
    }

    static double[] Numbers(string s, int expected)
    {
        if (expected < 1 || expected > 1_000_000) throw new ArgumentException("array_length");
        string[] tokens = s.Split(',');
        if (tokens.Length != expected) throw new ArgumentException("array_length");
        double[] xs = tokens.Select(t => double.Parse(t, CultureInfo.InvariantCulture)).ToArray();
        if (xs.Any(x => double.IsNaN(x) || double.IsInfinity(x))) throw new ArgumentException("nonfinite");
        return xs;
    }
    static int Integer(string s) => int.Parse(s, CultureInfo.InvariantCulture);
    static string Csv(double[] xs) => string.Join(",", xs.Select(x => x.ToString("G12", CultureInfo.InvariantCulture)));
    static string Evaluate(string request)
    {
        string[] p = request.Split('|');
        if (p.Length == 2 && p[0] == "PING") return $"PONG|{p[1]}|CSharp";
        if (p.Length == 8 && p[0] == "CONV")
        {
            int rows = Integer(p[2]), cols = Integer(p[3]), kr = Integer(p[4]), kc = Integer(p[5]);
            if (rows < 1 || cols < 1 || kr < 1 || kc < 1 || kr > rows || kc > cols)
                throw new ArgumentException("dims");
            double[] img = Numbers(p[6], checked(rows * cols));
            double[] kernel = Numbers(p[7], checked(kr * kc));
            int orows = rows - kr + 1, ocols = cols - kc + 1;
            double[] result = new double[checked(orows * ocols)];
            for (int r = 0; r < orows; r++)
                for (int c = 0; c < ocols; c++)
                {
                    double sum = 0;
                    for (int i = 0; i < kr; i++)
                        for (int j = 0; j < kc; j++)
                            sum += img[(r+i)*cols + c+j] * kernel[i*kc+j];
                    result[r*ocols + c] = sum;
                }
            return $"RESULT|{p[1]}|{orows}|{ocols}|{Csv(result)}";
        }
        if (p.Length == 10 && p[0] == "MLP")
        {
            int ni = Integer(p[2]), nh = Integer(p[3]), no = Integer(p[4]);
            if (ni < 1 || nh < 1 || no < 1 || (long)ni*nh > 1_000_000 || (long)nh*no > 1_000_000)
                throw new ArgumentException("dims");
            double[] input = Numbers(p[5], ni), w1 = Numbers(p[6], checked(nh*ni));
            double[] b1 = Numbers(p[7], nh), w2 = Numbers(p[8], checked(no*nh)), b2 = Numbers(p[9], no);
            double[] h = new double[nh], logits = new double[no];
            for (int i = 0; i < nh; i++)
            {
                double s = b1[i];
                for (int j = 0; j < ni; j++) s += w1[i*ni+j]*input[j];
                h[i] = Math.Max(0, s);
            }
            double maximum = double.NegativeInfinity;
            for (int i = 0; i < no; i++)
            {
                double s = b2[i];
                for (int j = 0; j < nh; j++) s += w2[i*nh+j]*h[j];
                logits[i] = s;
                maximum = Math.Max(maximum, s);
            }
            double total = 0;
            for (int i = 0; i < no; i++) { logits[i] = Math.Exp(logits[i] - maximum); total += logits[i]; }
            for (int i = 0; i < no; i++) logits[i] /= total;
            return $"RESULT|{p[1]}|1|{no}|{Csv(logits)}";
        }
        throw new ArgumentException("unsupported_command");
    }
}
