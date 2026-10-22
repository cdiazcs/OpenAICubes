"""Worker Python 3 con solo stdlib. Micro chunks CONV y MLP. Pipeline CNN completo."""
import math
import re
import socket
import struct
import sys
import threading
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from cnn_python.model import CNN, convolve_valid, relu, max_pooling

MAX_BYTES = 2_000_000
FRAME_TIMEOUT_SECONDS = 15.0
COMPUTE_THREADS = 1
COMPUTE_SLOTS = threading.BoundedSemaphore(COMPUTE_THREADS)
WEIGHTS_FILE = PROJECT_ROOT / "model" / "weights" / "weights.txt"

# Variable global para guardar los pesos en memoria
CNN_WEIGHTS = None
CNN_MODEL = None
_MODEL_LOCK = threading.Lock()
_DECIMAL = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')

def recv_exact(sock, count, deadline=None):
    parts = bytearray()
    while len(parts) < count:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('frame_timeout')
            sock.settimeout(remaining)
        more = sock.recv(count - len(parts))
        if not more:
            if len(parts) == 0:
                return None
            raise ConnectionError("truncated frame")
        parts.extend(more)
    return bytes(parts)


def recv_frame(sock):
    deadline = time.monotonic() + FRAME_TIMEOUT_SECONDS
    header = recv_exact(sock, 4, deadline)
    if header is None:
        return None
    length = struct.unpack('!I', header)[0]
    if not 0 < length <= MAX_BYTES:
        raise ValueError('invalid_length')
    body = recv_exact(sock, length, deadline)
    if body is None:
        raise ValueError('missing_frame')
    return body.decode('utf-8')


def send_frame(sock, message):
    data = message.encode('utf-8')
    if len(data) > MAX_BYTES:
        raise ValueError('response_too_large')
    sock.sendall(struct.pack('!I', len(data)) + data)


def numeric(raw, count):
    xs = raw.split(',')
    if not 0 < count <= 1_000_000 or len(xs) != count:
        raise ValueError('array_length')
    v = [float(x) for x in xs]
    if not all(math.isfinite(x) for x in v):
        raise ValueError('nonfinite')
    return v


def csv(v):
    return ','.join(format(x, '.12g') for x in v)


def finite(value):
    if not math.isfinite(value):
        raise ValueError('nonfinite_result')
    return value

def load_weights():
    """Carga validada y atomica; la ruta no depende del directorio de ejecucion."""
    global CNN_MODEL, CNN_WEIGHTS
    if CNN_MODEL is None:
        with _MODEL_LOCK:
            if CNN_MODEL is None:
                model = CNN.load(WEIGHTS_FILE)
                CNN_WEIGHTS = model.weights
                CNN_MODEL = model
    return CNN_MODEL


def run_full_cnn(image_csv):
    """Interfaz compatible IDENTIFY; calculo compartido con la CNN secuencial."""
    pixels = numeric(image_csv, 256)
    return list(load_weights().predict(pixels).probabilities)


def valid_id(value):
    return (bool(value) and len(value) <= 128
            and not any(ch == '|' or ch.isspace() or unicodedata.category(ch) in ('Cc', 'Cs') for ch in value))


def job(fields):
    """JOB req task CONV rows cols kr kc imageCSV kernelCSV (separado por |), o
    JOB req task DENSE nInputs nOutputs inputCSV weightsCSV biasCSV.
    PART|req|task|rows|cols|CSV17digitos|computeNanos; ERROR|req|task|reason.
    DENSE devuelve sum(W*x)+b sin activacion. CONV usa el helper de referencia.
    """
    request_id = fields[1] if len(fields) > 1 and valid_id(fields[1]) else 'unknown'
    task_id = fields[2] if len(fields) > 2 and valid_id(fields[2]) else 'unknown'

    def dimension(raw):
        if re.fullmatch(r'[0-9]{1,7}', raw) is None or not 1 <= int(raw) <= 1_000_000:
            raise ValueError('invalid_dimensions')
        return int(raw)

    def product(a, b):
        if a*b > 1_000_000:
            raise ValueError('invalid_dimensions')
        return a*b

    def numbers(raw, count):
        tokens = raw.split(',')
        if len(tokens) != count or any(len(token) > 64 or _DECIMAL.fullmatch(token) is None for token in tokens):
            raise ValueError('invalid_numbers')
        values = [float(token) for token in tokens]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('invalid_numbers')
        return values

    try:
        if len(fields) < 3 or not valid_id(fields[1]) or not valid_id(fields[2]):
            raise ValueError('invalid_job_id')
        if len(fields) < 4:
            raise ValueError('invalid_job_fields')
        if fields[3] == 'CONV':
            if len(fields) != 10:
                raise ValueError('invalid_job_fields')
            rows, cols, kr, kc = map(dimension, fields[4:8])
            if kr > rows or kc > cols or (rows-kr+1)*(cols-kc+1)*kr*kc > 10_000_000:
                raise ValueError('invalid_dimensions')
            image, kernel = numbers(fields[8], product(rows, cols)), numbers(fields[9], product(kr, kc))
            started = time.perf_counter_ns()
            result, out_rows, out_cols = convolve_valid(image, rows, cols, kernel, kr, kc)
        elif fields[3] == 'POST':
            if len(fields) != 8:
                raise ValueError('invalid_job_fields')
            rows, cols = dimension(fields[5]), dimension(fields[6])
            values = numbers(fields[7], product(rows, cols))
            started = time.perf_counter_ns()
            if fields[4] == 'POOL':
                if min(rows, cols) < 2:
                    raise ValueError('invalid_dimensions')
                result, out_rows, out_cols = max_pooling(relu(values), rows, cols)
            elif fields[4] == 'RELU':
                result, out_rows, out_cols = relu(values), rows, cols
            elif fields[4] == 'SOFTMAX':
                if rows != 1:
                    raise ValueError('invalid_dimensions')
                maximum = max(values)
                result = [math.exp(value-maximum) for value in values]
                total = sum(result)
                result, out_rows, out_cols = [value/total for value in result], rows, cols
            else:
                raise ValueError('unsupported_operation')
        elif fields[3] == 'DENSE':
            if len(fields) != 9:
                raise ValueError('invalid_job_fields')
            ni, no = dimension(fields[4]), dimension(fields[5])
            weight_count = product(ni, no)
            x, weights, bias = numbers(fields[6], ni), numbers(fields[7], weight_count), numbers(fields[8], no)
            started = time.perf_counter_ns()
            result = [finite(sum(weights[i*ni+j]*x[j] for j in range(ni))+bias[i]) for i in range(no)]
            out_rows, out_cols = 1, no
        else:
            raise ValueError('unsupported_operation')
        elapsed = max(0, time.perf_counter_ns()-started)
        return f'PART|{request_id}|{task_id}|{out_rows}|{out_cols}|' + ','.join(format(x, '.17g') for x in result) + f'|{elapsed}'
    except (ValueError, OverflowError) as error:
        code = str(error)
        if code not in ('invalid_job_id', 'invalid_job_fields', 'invalid_dimensions',
                        'invalid_numbers', 'nonfinite_result', 'unsupported_operation'):
            code = 'invalid_job'
        return f'ERROR|{request_id}|{task_id}|{code}'


def identify(fields):
    """RESULT conserva tres probabilidades: argmax = circulo/cruz/triangulo.

    Errores estables: invalid_request_id, invalid_identify_fields,
    invalid_image_size, invalid_pixels, model_unavailable, cnn_processing_error.
    IDs: 1..128 caracteres Unicode, sin separadores, espacios ni controles.
    Numeros: decimal ASCII, hasta 64 caracteres por pixel, finitos en [0,1].
    """
    job_id = fields[1] if len(fields) > 1 else ''
    if not valid_id(job_id):
        return 'ERROR|unknown|invalid_request_id'
    prefix = 'ERROR|' + job_id + '|'
    if len(fields) == 4 and fields[3] == 'SEQUENTIAL':
        fields = fields[:3]
    elif len(fields) == 4 and fields[3] == 'DISTRIBUTED':
        return prefix + 'unsupported_mode'
    if len(fields) != 3:
        return prefix + 'invalid_identify_fields'
    tokens = fields[2].split(',')
    if len(tokens) != 256:
        return prefix + 'invalid_image_size'
    if any(len(token) > 64 or _DECIMAL.fullmatch(token) is None for token in tokens):
        return prefix + 'invalid_pixels'
    pixels = [float(token) for token in tokens]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in pixels):
        return prefix + 'invalid_pixels'
    try:
        model = load_weights()
    except (OSError, ValueError, OverflowError):
        return prefix + 'model_unavailable'
    try:
        prediction = model.predict(pixels)
    except (ValueError, ArithmeticError):
        return prefix + 'cnn_processing_error'
    return 'RESULT|' + job_id + '|1|3|' + csv(prediction.probabilities)


def evaluate(message):
    p = message.split('|')
    if p[0] == 'JOB':
        return job(p)
    if p[0] == 'CAPS':
        if len(p) != 2 or not valid_id(p[1]):
            return 'ERROR|unknown|invalid_request_id'
        return f'CAPS|{p[1]}|Python|CONV,DENSE,POST,IDENTIFY|{COMPUTE_THREADS}'
    if p[0] == 'IDENTIFY':
        return identify(p)
    if len(p) == 2 and p[0] == 'PING':
        return f'PONG|{p[1]}|Python'
        
    if len(p) == 8 and p[0] == 'CONV':
        rows, cols, kr, kc = map(int, p[2:6])
        if min(rows, cols, kr, kc) <= 0 or kr > rows or kc > cols:
            raise ValueError('dims')
        image = numeric(p[6], rows * cols)
        kernel = numeric(p[7], kr * kc)
        
        result, out_rows, out_cols = convolve_valid(image, rows, cols, kernel, kr, kc)
        return f'RESULT|{p[1]}|{out_rows}|{out_cols}|{csv(result)}'
        
    if len(p) == 10 and p[0] == 'MLP':
        ni, nh, no = map(int, p[2:5])
        if min(ni, nh, no) <= 0 or ni * nh > 1_000_000 or nh * no > 1_000_000:
            raise ValueError('dims')
        x = numeric(p[5], ni)
        w1, b1 = numeric(p[6], ni * nh), numeric(p[7], nh)
        w2, b2 = numeric(p[8], nh * no), numeric(p[9], no)
        
        h = [max(0.0, finite(sum(w1[i * ni+j] * x[j] for j in range(ni)) + b1[i])) for i in range(nh)]
        logits = [finite(sum(w2[i * nh+j] * h[j] for j in range(nh)) + b2[i]) for i in range(no)]
        exps = [math.exp(y - max(logits)) for y in logits]
        total = sum(exps)
        return f'RESULT|{p[1]}|1|{no}|{csv([x / total for x in exps])}'

    raise ValueError('unsupported_command')


def handle(conn, address):
    with conn:
        conn.settimeout(FRAME_TIMEOUT_SECONDS)
        try:
            while True:
                request = recv_frame(conn)
                if request is None:
                    return
                try:
                    if request.partition('|')[0] in ('JOB', 'CONV', 'MLP', 'IDENTIFY'):
                        if not COMPUTE_SLOTS.acquire(timeout=FRAME_TIMEOUT_SECONDS):
                            parts = request.split('|')
                            if parts[0] == 'JOB':
                                ids = '|'.join(parts[i] if len(parts) > i and valid_id(parts[i]) else 'unknown' for i in (1, 2))
                            else:
                                ids = parts[1] if len(parts) > 1 else 'unknown'
                            answer = 'ERROR|' + ids + '|worker_busy'
                        else:
                            try:
                                answer = evaluate(request)
                            finally:
                                COMPUTE_SLOTS.release()
                    else:
                        answer = evaluate(request)
                except (ValueError, OverflowError) as ex:
                    items = request.split('|')
                    job_id = items[1] if len(items) > 1 else 'unknown'
                    answer = f'ERROR|{job_id}|{type(ex).__name__}'
                conn.settimeout(FRAME_TIMEOUT_SECONDS)
                send_frame(conn, answer)
        except (OSError, ValueError) as ex:
            print(f'conexion {address} terminada: {ex}', file=sys.stderr)


def main():
    global COMPUTE_THREADS, COMPUTE_SLOTS
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9102
    COMPUTE_THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if not 1 <= COMPUTE_THREADS <= 256:
        raise ValueError('invalid_threads')
    COMPUTE_SLOTS = threading.BoundedSemaphore(COMPUTE_THREADS)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(64)
    print(f'WorkerPython escuchando en {port}, compute_threads={COMPUTE_THREADS} (GIL: no garantiza paralelismo CPU)', flush=True)
    with server:
        while True:
            conn, addr = server.accept()
            threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == '__main__':
    main()
