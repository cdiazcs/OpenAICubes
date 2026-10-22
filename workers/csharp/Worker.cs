// Worker C# (.NET 8): sin NuGet ni librerías externas.
using System;
using System.Buffers.Binary;
using System.Globalization;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

static class Worker
{
    private const int MaxBytes = 2_000_000;
    private static int ComputeThreads = 4;
    private static SemaphoreSlim ComputeSlots = new SemaphoreSlim(4, 4);
    private static readonly UTF8Encoding Utf8 = new UTF8Encoding(false, true);
    private static readonly Regex Decimal = new Regex(@"\A[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\z", RegexOptions.CultureInvariant);
    static async Task Main(string[] args)
    {
        int port = args.Length > 0 ? int.Parse(args[0], CultureInfo.InvariantCulture) : 9103;
        ComputeThreads = args.Length > 1 ? int.Parse(args[1], CultureInfo.InvariantCulture) : 4;
        if (ComputeThreads < 1 || ComputeThreads > 256) throw new ArgumentException("invalid_threads");
        ComputeSlots.Dispose();
        ComputeSlots = new SemaphoreSlim(ComputeThreads, ComputeThreads);
        var listener = new TcpListener(IPAddress.Any, port);
        listener.Start();
        Console.WriteLine($"WorkerCSharp escuchando en {port}, compute_threads={ComputeThreads}");
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
                    try
                    {
                        string command = request.Split('|', 2)[0];
                        bool limited = command == "JOB" || command == "CONV" || command == "MLP";
                        if (limited && !ComputeSlots.Wait(15000))
                        {
                            string[] fields = request.Split('|');
                            string ids = command == "JOB" ? string.Join("|", new[] { 1, 2 }.Select(
                                i => fields.Length > i && ValidId(fields[i]) ? fields[i] : "unknown"))
                                : (fields.Length > 1 ? fields[1] : "unknown");
                            response = "ERROR|" + ids + "|worker_busy";
                        }
                        else
                        {
                            try { response = Evaluate(request); }
                            finally { if (limited) ComputeSlots.Release(); }
                        }
                    }
                    catch (Exception ex) when (ex is FormatException || ex is ArgumentException || ex is OverflowException)
                    {
                        string[] parts = request.Split('|');
                        string id = parts.Length > 1 ? parts[1] : "unknown";
                        response = $"ERROR|{id}|{ex.GetType().Name}";
                    }
                    WriteFrame(stream, response);
                }
            }
            catch (Exception ex) when (ex is IOException || ex is DecoderFallbackException)
            { Console.Error.WriteLine($"conexion cerrada: {ex.Message}"); }
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
        return Utf8.GetString(data);
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
        byte[] payload = Utf8.GetBytes(text);
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
    static double Finite(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            throw new ArgumentException("nonfinite_result");
        return value;
    }
    static string Csv(double[] xs) => string.Join(",", xs.Select(x => x.ToString("G12", CultureInfo.InvariantCulture)));
    static string Evaluate(string request)
    {
        string[] p = request.Split('|');
        if (p[0] == "CAPS")
        {
            if (p.Length != 2 || !ValidId(p[1])) return "ERROR|unknown|invalid_request_id";
            return $"CAPS|{p[1]}|CSharp|CONV,DENSE,POST|{ComputeThreads}";
        }
        if (p[0] == "JOB") return Job(p);
        if (p.Length == 2 && p[0] == "PING") return $"PONG|{p[1]}|CSharp";
        if (p.Length == 8 && p[0] == "CONV")
        {
            int rows = Integer(p[2]), cols = Integer(p[3]), kr = Integer(p[4]), kc = Integer(p[5]);
            if (rows < 1 || cols < 1 || kr < 1 || kc < 1 || kr > rows || kc > cols)
                throw new ArgumentException("dims");
            double[] img = Numbers(p[6], checked(rows * cols));
            double[] kernel = Numbers(p[7], checked(kr * kc));
            int orows = rows - kr + 1, ocols = cols - kc + 1;
            double[] result = Convolution(img, rows, cols, kernel, kr, kc);
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
                h[i] = Math.Max(0, Finite(s));
            }
            double maximum = double.NegativeInfinity;
            for (int i = 0; i < no; i++)
            {
                double s = b2[i];
                for (int j = 0; j < nh; j++) s += w2[i*nh+j]*h[j];
                logits[i] = Finite(s);
                maximum = Math.Max(maximum, s);
            }
            double total = 0;
            for (int i = 0; i < no; i++) { logits[i] = Math.Exp(logits[i] - maximum); total += logits[i]; }
            for (int i = 0; i < no; i++) logits[i] /= total;
            return $"RESULT|{p[1]}|1|{no}|{Csv(logits)}";
        }
        throw new ArgumentException("unsupported_command");
    }

    // CAPS|req -> CAPS|req|CSharp|CONV,DENSE|computeThreads.
    // JOB|req|task|CONV|rows|cols|kr|kc|imageCSV|kernelCSV (correlacion valida).
    // JOB|req|task|DENSE|nInputs|nOutputs|inputCSV|weightsCSV|biasCSV (sin activacion).
    // PART|req|task|rows|cols|valuesCSV(G17)|computeNanos; ERROR|req|task|reason.
    static string Job(string[] p)
    {
        string requestId = p.Length > 1 && ValidId(p[1]) ? p[1] : "unknown";
        string taskId = p.Length > 2 && ValidId(p[2]) ? p[2] : "unknown";
        try
        {
            if (p.Length < 3 || !ValidId(p[1]) || !ValidId(p[2])) throw new ArgumentException("invalid_job_id");
            if (p.Length < 4) throw new ArgumentException("invalid_job_fields");
            int rows, cols;
            double[] result;
            long started;
            if (p[3] == "CONV")
            {
                if (p.Length != 10) throw new ArgumentException("invalid_job_fields");
                int ir = Dimension(p[4]), ic = Dimension(p[5]), kr = Dimension(p[6]), kc = Dimension(p[7]);
                if (kr > ir || kc > ic) throw new ArgumentException("invalid_dimensions");
                rows = ir-kr+1; cols = ic-kc+1;
                if ((long)rows*cols*kr*kc > 10_000_000) throw new ArgumentException("invalid_dimensions");
                double[] image = JobNumbers(p[8], Product(ir, ic)), kernel = JobNumbers(p[9], Product(kr, kc));
                started = Stopwatch.GetTimestamp();
                result = Convolution(image, ir, ic, kernel, kr, kc);
            }
            else if (p[3] == "POST")
            {
                if (p.Length != 8) throw new ArgumentException("invalid_job_fields");
                int ir = Dimension(p[5]), ic = Dimension(p[6]);
                double[] x = JobNumbers(p[7], Product(ir, ic));
                rows = ir; cols = ic; started = Stopwatch.GetTimestamp();
                if (p[4] == "POOL")
                {
                    if (ir < 2 || ic < 2) throw new ArgumentException("invalid_dimensions");
                    rows = ir/2; cols = ic/2; result = new double[rows*cols];
                    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
                        int at = 2*r*ic+2*c;
                        result[r*cols+c] = Math.Max(0, Math.Max(Math.Max(x[at], x[at+1]), Math.Max(x[at+ic], x[at+ic+1])));
                    }
                }
                else if (p[4] == "RELU") { result = x; for (int i = 0; i < x.Length; i++) result[i] = Math.Max(0, x[i]); }
                else if (p[4] == "SOFTMAX") {
                    if (ir != 1) throw new ArgumentException("invalid_dimensions");
                    double max = x.Max(), total = 0; result = x;
                    for (int i = 0; i < x.Length; i++) { result[i] = Math.Exp(x[i]-max); total += result[i]; }
                    for (int i = 0; i < x.Length; i++) result[i] /= total;
                }
                else throw new ArgumentException("unsupported_operation");
            }
            else if (p[3] == "DENSE")
            {
                if (p.Length != 9) throw new ArgumentException("invalid_job_fields");
                int ni = Dimension(p[4]);
                rows = 1; cols = Dimension(p[5]);
                int weightCount = Product(ni, cols);
                double[] x = JobNumbers(p[6], ni), weights = JobNumbers(p[7], weightCount), bias = JobNumbers(p[8], cols);
                started = Stopwatch.GetTimestamp();
                result = new double[cols];
                for (int i = 0; i < cols; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < ni; j++) sum += weights[i*ni+j]*x[j];
                    result[i] = Finite(sum+bias[i]);
                }
            }
            else throw new ArgumentException("unsupported_operation");
            long elapsed = Math.Max(0, (long)((Stopwatch.GetTimestamp()-started)*(1_000_000_000.0/Stopwatch.Frequency)));
            string values = string.Join(",", result.Select(x => x.ToString("G17", CultureInfo.InvariantCulture)));
            return $"PART|{requestId}|{taskId}|{rows}|{cols}|{values}|{elapsed}";
        }
        catch (Exception ex) when (ex is ArgumentException || ex is FormatException || ex is OverflowException)
        {
            string code = ex.Message;
            if (!new[] { "invalid_job_id", "invalid_job_fields", "invalid_dimensions", "invalid_numbers",
                         "nonfinite_result", "unsupported_operation" }.Contains(code)) code = "invalid_job";
            return $"ERROR|{requestId}|{taskId}|{code}";
        }
    }

    static int Dimension(string raw)
    {
        if (!Regex.IsMatch(raw, @"\A[0-9]{1,7}\z")) throw new ArgumentException("invalid_dimensions");
        int value = Integer(raw);
        if (value < 1 || value > 1_000_000) throw new ArgumentException("invalid_dimensions");
        return value;
    }
    static int Product(int a, int b)
    {
        if ((long)a*b > 1_000_000) throw new ArgumentException("invalid_dimensions");
        return a*b;
    }
    static double[] JobNumbers(string raw, int count)
    {
        string[] tokens = raw.Split(',');
        if (tokens.Length != count) throw new ArgumentException("invalid_numbers");
        double[] values = new double[count];
        for (int i = 0; i < count; i++)
        {
            if (tokens[i].Length > 64 || !Decimal.IsMatch(tokens[i])
                || !double.TryParse(tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture, out values[i])
                || !double.IsFinite(values[i])) throw new ArgumentException("invalid_numbers");
        }
        return values;
    }
    static bool ValidId(string id)
    {
        if (id.Length == 0) return false;
        int count = 0;
        for (int i = 0; i < id.Length; i++)
        {
            char c = id[i];
            if (c == '|' || char.IsControl(c) || char.IsWhiteSpace(c)) return false;
            if (char.IsSurrogate(c))
            {
                if (!char.IsHighSurrogate(c) || i+1 >= id.Length || !char.IsLowSurrogate(id[i+1])) return false;
                i++;
            }
            if (++count > 128) return false;
        }
        return true;
    }
    static double[] Convolution(double[] image, int rows, int cols, double[] kernel, int kr, int kc)
    {
        int orows = rows-kr+1, ocols = cols-kc+1;
        double[] result = new double[orows*ocols];
        for (int r = 0; r < orows; r++) for (int c = 0; c < ocols; c++)
        {
            double sum = 0;
            for (int i = 0; i < kr; i++) for (int j = 0; j < kc; j++)
                sum += image[(r+i)*cols+c+j]*kernel[i*kc+j];
            result[r*ocols+c] = Finite(sum);
        }
        return result;
    }
}
