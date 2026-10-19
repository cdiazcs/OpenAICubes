"""Prueba de interoperabilidad contra uno o más workers o servidor central.

Uso:
    python tests/smoke_test.py 127.0.0.1:9101 127.0.0.1:9102 127.0.0.1:9000
"""
import math
import socket
import struct
import sys

TESTS = [
    ("PING|t0", "PONG"),
    ("CONV|t1|3|3|2|2|1,2,3,4,5,6,7,8,9|1,0,0,-1", "CONV"),
    ("MLP|t2|2|2|2|1,2|1,0,0,1|0,0|1,0,0,1|0,0", "MLP"),
]


def receive_exact(s, n):
    result = b''
    while len(result) < n:
        part = s.recv(n-len(result))
        if not part:
            raise RuntimeError('conexion interrumpida')
        result += part
    return result


def exchange(host, port, message):
    with socket.create_connection((host, port), timeout=5) as s:
        s.settimeout(10)
        data = message.encode('utf-8')
        s.sendall(struct.pack('!I', len(data)) + data)
        size = struct.unpack('!I', receive_exact(s, 4))[0]
        if size <= 0 or size > 2_000_000:
            raise ValueError('invalid size')
        return receive_exact(s, size).decode('utf-8')


def check(reply, kind):
    fields = reply.split('|')
    if kind == 'PONG':
        assert fields[0] == 'PONG' and fields[1] == 't0', reply
    elif kind == 'CONV':
        assert fields[:4] == ['RESULT', 't1', '2', '2'], reply
        vals = [float(x) for x in fields[4].split(',')]
        assert len(vals) == 4 and all(abs(v + 4) < 1e-8 for v in vals), reply
    elif kind == 'MLP':
        assert fields[:4] == ['RESULT', 't2', '1', '2'], reply
        vals = [float(x) for x in fields[4].split(',')]
        expected = [1/(1+math.e), math.e/(1+math.e)]
        assert len(vals) == 2 and all(abs(v-e) < 1e-8 for v, e in zip(vals, expected)), reply


def main():
    endpoints = sys.argv[1:] or ['127.0.0.1:9101', '127.0.0.1:9102']
    for endpoint in endpoints:
        host, port = endpoint.rsplit(':', 1)
        for message, kind in TESTS:
            response = exchange(host, int(port), message)
            check(response, kind)
            print(f'{endpoint} {kind}: OK -> {response[:100]}')
    print('TODAS LAS PRUEBAS: OK')


if __name__ == '__main__':
    main()
