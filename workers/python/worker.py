"""Worker Python 3 con solo stdlib. Micro chunks CONV y MLP. Pipeline CNN completo."""
import math
import socket
import struct
import sys
import threading
import os

MAX_BYTES = 2_000_000
WEIGHTS_FILE = "model/weights/weights.txt" # Ruta relativa desde donde se ejecute

# Variable global para guardar los pesos en memoria
CNN_WEIGHTS = None

def recv_exact(sock, count):
    parts = bytearray()
    while len(parts) < count:
        more = sock.recv(count - len(parts))
        if not more:
            if len(parts) == 0:
                return None
            raise ConnectionError("truncated frame")
        parts.extend(more)
    return bytes(parts)


def recv_frame(sock):
    header = recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack('!I', header)[0]
    if not 0 < length <= MAX_BYTES:
        raise ValueError('invalid_length')
    body = recv_exact(sock, length)
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

# ==========================================
# NUEVO: LÓGICA DE LA CNN COMPLETA
# ==========================================

def load_weights():
    """Carga los pesos desde el archivo de texto si no están en memoria."""
    global CNN_WEIGHTS
    if CNN_WEIGHTS is not None:
        return
    
    if not os.path.exists(WEIGHTS_FILE):
        print(f"[ADVERTENCIA] No se encontró {WEIGHTS_FILE}.", file=sys.stderr)
        CNN_WEIGHTS = {}
        return

    CNN_WEIGHTS = {}
    current_key = None
    
    try:
        with open(WEIGHTS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line in ['FILTER1', 'FILTER2', 'W1', 'B1', 'W2', 'B2']:
                    current_key = line
                elif current_key:
                    # Convertir la línea de CSV a lista de floats
                    CNN_WEIGHTS[current_key] = [float(x) for x in line.split(',')]
                    current_key = None
        print("[INFO] Pesos cargados correctamente en memoria.", flush=True)
    except Exception as e:
        print(f"[ERROR] Fallo al cargar pesos: {e}", file=sys.stderr)
        CNN_WEIGHTS = {}

def convolve_valid(image, rows, cols, kernel, kr, kc):
    """Helper: Correlación válida para una imagen y un filtro (kernel)."""
    out_rows, out_cols = rows - kr + 1, cols - kc + 1
    result = []
    for r in range(out_rows):
        for c in range(out_cols):
            acc = 0.0
            for i in range(kr):
                for j in range(kc):
                    acc += image[(r+i)*cols + (c+j)] * kernel[i*kc+j]
            result.append(acc)
    return result, out_rows, out_cols

def relu(matrix):
    """Aplica ReLU a una lista plana."""
    return [max(0.0, x) for x in matrix]

def max_pooling(matrix, rows, cols):
    """Max Pooling 2x2, stride 2."""
    out_rows = rows // 2
    out_cols = cols // 2
    result = []
    for r in range(out_rows):
        for c in range(out_cols):
            # Extraer ventana 2x2
            p1 = matrix[(r*2)*cols + (c*2)]
            p2 = matrix[(r*2)*cols + (c*2 + 1)]
            p3 = matrix[(r*2 + 1)*cols + (c*2)]
            p4 = matrix[(r*2 + 1)*cols + (c*2 + 1)]
            result.append(max(p1, p2, p3, p4))
    return result, out_rows, out_cols

def run_full_cnn(image_csv):
    """Pipeline: Conv -> ReLU -> Pooling -> Flatten -> MLP -> Softmax."""
    load_weights()
    if not CNN_WEIGHTS:
        raise ValueError('weights_not_loaded')

    # 1. Parsear imagen 16x16 (256 valores)
    image = numeric(image_csv, 256)
    # Nota: Si la imagen viene en escala 0-255, normalízala aquí (x/255.0)

    # 2. Capa Convolucional (2 filtros de 3x3) -> 14x14 (CORREGIDO A MAYÚSCULAS)
    conv1, r1, c1 = convolve_valid(image, 16, 16, CNN_WEIGHTS['FILTER1'], 3, 3)
    conv2, r2, c2 = convolve_valid(image, 16, 16, CNN_WEIGHTS['FILTER2'], 3, 3)

    # 3. ReLU
    relu1 = relu(conv1)
    relu2 = relu(conv2)

    # 4. Max Pooling -> 7x7
    pool1, pr1, pc1 = max_pooling(relu1, r1, c1)
    pool2, pr2, pc2 = max_pooling(relu2, r2, c2)

    # 5. Flatten (98 componentes)
    flat = pool1 + pool2 # 49 + 49 = 98

    # 6. MLP (98 -> 8 -> 3) (CORREGIDO A MAYÚSCULAS)
    ni, nh, no = 98, 8, 3
    w1, b1 = CNN_WEIGHTS['W1'], CNN_WEIGHTS['B1']
    w2, b2 = CNN_WEIGHTS['W2'], CNN_WEIGHTS['B2']

    h = [max(0.0, sum(w1[i * ni+j] * flat[j] for j in range(ni)) + b1[i]) for i in range(nh)]
    logits = [sum(w2[i * nh+j] * h[j] for j in range(nh)) + b2[i] for i in range(no)]

    # 7. Softmax
    exps = [math.exp(y - max(logits)) for y in logits]
    total = sum(exps)
    return [x / total for x in exps]

# ==========================================
# EVALUATE ACTUALIZADO
# ==========================================

def evaluate(message):
    p = message.split('|')
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
        
        h = [max(0.0, sum(w1[i * ni+j] * x[j] for j in range(ni)) + b1[i]) for i in range(nh)]
        logits = [sum(w2[i * nh+j] * h[j] for j in range(nh)) + b2[i] for i in range(no)]
        exps = [math.exp(y - max(logits)) for y in logits]
        total = sum(exps)
        return f'RESULT|{p[1]}|1|{no}|{csv([x / total for x in exps])}'

    # NUEVO COMANDO: IDENTIFY
    if len(p) == 3 and p[0] == 'IDENTIFY':
        request_id = p[1]
        image_csv = p[2]
        
        try:
            probs = run_full_cnn(image_csv)
            return f'RESULT|{request_id}|1|3|{csv(probs)}'
        except Exception as ex:
            raise ValueError(f'cnn_error_{type(ex).__name__}')

    raise ValueError('unsupported_command')


def handle(conn, address):
    with conn:
        conn.settimeout(15)
        try:
            while True:
                request = recv_frame(conn)
                if request is None:
                    return
                try:
                    answer = evaluate(request)
                except (ValueError, OverflowError) as ex:
                    items = request.split('|')
                    job_id = items[1] if len(items) > 1 else 'unknown'
                    answer = f'ERROR|{job_id}|{type(ex).__name__}'
                send_frame(conn, answer)
        except (OSError, ConnectionError, UnicodeError) as ex:
            print(f'conexion {address} terminada: {ex}', file=sys.stderr)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9102
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(64)
    print(f'WorkerPython escuchando en {port}', flush=True)
    with server:
        while True:
            conn, addr = server.accept()
            threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == '__main__':
    main()