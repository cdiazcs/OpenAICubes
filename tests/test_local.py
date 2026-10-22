"""Comprueba el transporte del IDENTIFY experimental; no mide precision de clases."""
import argparse
import math
import random
import socket
import struct
import sys

from smoke_test import receive_exact

MAX_BYTES = 2_000_000


def send_msg(sock, msg):
    payload = msg.encode('utf-8')
    if not 0 < len(payload) <= MAX_BYTES:
        raise ValueError('invalid_request_length')
    sock.sendall(struct.pack('!I', len(payload)) + payload)
    length = struct.unpack('!I', receive_exact(sock, 4))[0]
    if not 0 < length <= MAX_BYTES:
        raise ValueError('invalid_response_length')
    return receive_exact(sock, length).decode('utf-8')


def validate_identify(reply, request_id):
    fields = reply.split('|')
    if len(fields) != 5 or fields[:4] != ['RESULT', request_id, '1', '3']:
        raise ValueError('unexpected_response: ' + reply)
    probabilities = [float(value) for value in fields[4].split(',')]
    if (len(probabilities) != 3
            or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            or abs(sum(probabilities) - 1) > 1e-8):
        raise ValueError('invalid_probabilities: ' + reply)
    return probabilities


def test_identify(host='127.0.0.1', port=9102, seed=42):
    rng = random.Random(seed)
    image = ','.join(str(round(rng.uniform(0, 1), 2)) for _ in range(256))
    request_id = 'req_123'
    with socket.create_connection((host, port), timeout=5) as connection:
        connection.settimeout(10)
        reply = send_msg(connection, 'IDENTIFY|' + request_id + '|' + image)
    probabilities = validate_identify(reply, request_id)
    print('Respuesta:', reply)
    print('Suma de probabilidades:', sum(probabilities))
    print('OK: distribucion valida. Esta prueba no demuestra precision de clasificacion.')
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', nargs='?', default='127.0.0.1')
    parser.add_argument('port', nargs='?', type=int, default=9102)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        test_identify(args.host, args.port, args.seed)
    except (OSError, ValueError, RuntimeError) as error:
        print('ERROR:', error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
