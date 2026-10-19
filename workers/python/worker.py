"""Worker Python 3 con solo stdlib. Micro chunks CONV y MLP."""
import math
import socket
import struct
import sys
import threading

MAX_BYTES = 2_000_000


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
        out_rows, out_cols = rows - kr + 1, cols - kc + 1
        result = []
        for r in range(out_rows):
            for c in range(out_cols):
                result.append(sum(image[(r+i)*cols + (c+j)] * kernel[i*kc+j]
                                  for i in range(kr) for j in range(kc)))
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
