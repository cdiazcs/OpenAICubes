"""Auditoria TCP real, solo stdlib: python tests/test_protocol.py.

Compila fuentes y prueba Java/Python/C# y CentralServer en puertos locales libres.
Detiene solo los procesos propios; conserva binarios y logs en build/stage0-*.
"""
from concurrent.futures import ThreadPoolExecutor
import csv as csv_module
from datetime import datetime
import math
from pathlib import Path
import random
import socket
import struct
import subprocess
import sys
import time
import unittest
import uuid

from smoke_test import TESTS, check, exchange, receive_exact
from test_local import send_msg, validate_identify

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 2_000_000


def frame(text):
    payload = text.encode('utf-8')
    return struct.pack('!I', len(payload)) + payload


def receive(connection):
    size = struct.unpack('!I', receive_exact(connection, 4))[0]
    if not 0 < size <= MAX_BYTES:
        raise AssertionError('invalid response length: ' + str(size))
    return receive_exact(connection, size).decode('utf-8')


def csv(values):
    return ','.join(str(value) for value in values)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
        cls.artifacts = ROOT / 'build' / ('stage0-' + run_id)
        cls.artifacts.mkdir(parents=True)
        cls.processes, cls.logs, cls.ports = [], [], set()
        cls.addClassCleanup(cls.cleanup)
        print('\nArtefactos:', cls.artifacts, flush=True)
        cls.compile(['javac', '--release', '8', '-encoding', 'UTF-8', '-d',
                     str(cls.artifacts), 'server-java/CentralServer.java', 'server-java/DistributedCnn.java',
                     'clients-java/ClientCli.java', 'workers/java/WorkerJava.java',
                     'benchmarks/Benchmark.java', 'benchmarks/CnnBenchmark.java',
                     'benchmarks/LocalCnn.java', 'benchmarks/PhysicalConfig.java'], 'javac')
        cls.compile(['dotnet', 'build', 'workers/csharp/Worker.csproj', '--nologo',
                     '--verbosity', 'minimal', '-o', str(cls.artifacts / 'csharp')],
                    'dotnet-build')
        cls.endpoints = {}
        for language, command in [
            ('Java', ['java', '-cp', str(cls.artifacts), 'WorkerJava']),
            ('Python', [sys.executable, 'workers/python/worker.py']),
            ('CSharp', ['dotnet', str(cls.artifacts / 'csharp' / 'Worker.dll')]),
        ]:
            port = cls.free_port()
            cls.start(command + [str(port)], port, language)
            cls.endpoints[language] = port
        cls.central = cls.start_central('Central', list(cls.endpoints.values()))
        cls.endpoints['Central'] = cls.central

    @classmethod
    def compile(cls, command, name):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=90)
        (cls.artifacts / (name + '.log')).write_text(result.stdout + result.stderr,
                                                   encoding='utf-8')
        if result.returncode:
            raise RuntimeError(name + ' fallo:\n' + result.stdout + result.stderr)

    @classmethod
    def free_port(cls):
        while True:
            with socket.socket() as connection:
                connection.bind(('127.0.0.1', 0))
                port = connection.getsockname()[1]
            if port not in cls.ports:
                cls.ports.add(port)
                return port

    @classmethod
    def start(cls, command, port, name):
        log = (cls.artifacts / (name + '.log')).open('w', encoding='utf-8')
        cls.logs.append(log)
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        cls.processes.append(process)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(name + ' termino; revisar ' + str(log.name))
            try:
                if exchange('127.0.0.1', port, 'PING|ready').startswith('PONG|ready|'):
                    return process
            except OSError:
                time.sleep(.05)
        raise RuntimeError(name + ' no responde; revisar ' + str(log.name))

    @classmethod
    def start_central(cls, name, workers):
        port = cls.free_port()
        command = ['java', '-cp', str(cls.artifacts), 'CentralServer', str(port)]
        cls.start(command + ['127.0.0.1:' + str(p) for p in workers], port, name)
        return port

    @classmethod
    def cleanup(cls):
        for process in reversed(cls.processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(cls.processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in cls.logs:
            log.close()

    def result(self, reply, request_id, rows, cols, expected):
        fields = reply.split('|')
        self.assertEqual(len(fields), 5, reply)
        self.assertEqual(fields[:4], ['RESULT', request_id, str(rows), str(cols)], reply)
        values = [float(value) for value in fields[4].split(',')]
        self.assertEqual(len(values), rows * cols, reply)
        self.assertEqual(len(values), len(expected), reply)
        for actual, reference in zip(values, expected):
            self.assertTrue(math.isfinite(actual), reply)
            self.assertAlmostEqual(actual, reference, delta=1e-8)

    def error(self, reply, request_id):
        fields = reply.split('|')
        self.assertEqual(len(fields), 3, reply)
        self.assertEqual(fields[:2], ['ERROR', request_id], reply)
        self.assertTrue(fields[2], reply)

    def test_documented_operations(self):
        for name, port in self.endpoints.items():
            for message, kind in TESTS:
                with self.subTest(endpoint=name, command=kind):
                    check(exchange('127.0.0.1', port, message), kind)

    def test_ping_languages_and_round_robin(self):
        for language in ('Java', 'Python', 'CSharp'):
            self.assertEqual(exchange('127.0.0.1', self.endpoints[language], 'PING|id'),
                             'PONG|id|' + language)
        languages = {exchange('127.0.0.1', self.central, 'PING|rr').split('|')[2]
                     for _ in range(3)}
        self.assertEqual(languages, {'Java', 'Python', 'CSharp'})

    def test_numeric_reference(self):
        rng = random.Random(42)
        for case in range(8):
            rows, cols, kr, kc = 4, 5, 2, 3
            image = [rng.uniform(-2, 2) for _ in range(rows * cols)]
            kernel = [rng.uniform(-1, 1) for _ in range(kr * kc)]
            expected = [sum(image[(r+i)*cols+c+j] * kernel[i*kc+j]
                            for i in range(kr) for j in range(kc))
                        for r in range(rows-kr+1) for c in range(cols-kc+1)]
            request_id = 'conv' + str(case)
            message = '|'.join(['CONV', request_id, str(rows), str(cols), str(kr),
                                str(kc), csv(image), csv(kernel)])
            for name, port in self.endpoints.items():
                with self.subTest(endpoint=name, convolution=case):
                    self.result(exchange('127.0.0.1', port, message), request_id, 3, 3, expected)

            ni, nh, no = 4, 3, 2
            x = [rng.uniform(-2, 2) for _ in range(ni)]
            w1 = [rng.uniform(-1, 1) for _ in range(ni*nh)]
            b1 = [rng.uniform(-1, 1) for _ in range(nh)]
            w2 = [rng.uniform(-1, 1) for _ in range(nh*no)]
            b2 = [rng.uniform(-1, 1) for _ in range(no)]
            hidden = [max(0, sum(w1[i*ni+j]*x[j] for j in range(ni)) + b1[i])
                      for i in range(nh)]
            logits = [sum(w2[i*nh+j]*hidden[j] for j in range(nh)) + b2[i]
                      for i in range(no)]
            exps = [math.exp(value-max(logits)) for value in logits]
            expected = [value/sum(exps) for value in exps]
            request_id = 'mlp' + str(case)
            message = '|'.join(['MLP', request_id, str(ni), str(nh), str(no)]
                               + [csv(v) for v in (x, w1, b1, w2, b2)])
            for name, port in self.endpoints.items():
                with self.subTest(endpoint=name, mlp=case):
                    self.result(exchange('127.0.0.1', port, message), request_id, 1, 2, expected)

    def test_relu_and_stable_softmax(self):
        message = 'MLP|stable|3|2|3|-1,2,0.5|1,-1,0,0,1,2|0,-0.5|1,0,0,1,1,1|1000,1000,-1000'
        exps = [math.exp(-2.5), 1, math.exp(-2000)]
        for name, port in self.endpoints.items():
            with self.subTest(endpoint=name):
                self.result(exchange('127.0.0.1', port, message), 'stable', 1, 3,
                            [value/sum(exps) for value in exps])

    def test_invalid_messages_and_connection_reuse(self):
        messages = [
            ('UNKNOWN|bad', 'bad'), ('PING|extra|field', 'extra'),
            ('CONV|dims|0|3|1|1|1,2,3|1', 'dims'),
            ('CONV|kernel|2|2|3|1|1,2,3,4|1,1,1', 'kernel'),
            ('CONV|count|2|2|1|1|1,2,3|1', 'count'),
            ('CONV|nan|1|1|1|1|NaN|1', 'nan'),
            ('CONV|inf|1|1|1|1|Infinity|1', 'inf'),
            ('CONV|huge|2147483647|2147483647|1|1|1|1', 'huge'),
            ('MLP|dims|1|0|1|1|1|0|1|0', 'dims'),
            ('MLP|count|2|1|1|1|1,1|0|1|0', 'count'),
        ]
        for name, port in self.endpoints.items():
            with socket.create_connection(('127.0.0.1', port), timeout=5) as connection:
                connection.settimeout(10)
                for message, request_id in messages:
                    with self.subTest(endpoint=name, request=request_id):
                        connection.sendall(frame(message))
                        self.error(receive(connection), request_id)
                connection.sendall(frame('PING|recovered'))
                self.assertTrue(receive(connection).startswith('PONG|recovered|'))

    def test_nonfinite_computation_rejected(self):
        messages = [
            'CONV|overflow|1|1|1|1|1e308|2',
            'MLP|overflow|1|1|2|1e308|2|0|1,1|0,0',
            'MLP|overflow|1|1|2|1|1|0|1e308,1e308|1e308,1e308',
            'MLP|overflow|1|1|2|1e308|-2|0|1,1|0,0',
        ]
        for name, port in self.endpoints.items():
            for message in messages:
                with self.subTest(endpoint=name, message=message):
                    self.error(exchange('127.0.0.1', port, message), 'overflow')

    def test_fragmented_and_coalesced_frames(self):
        for name, port in self.endpoints.items():
            with self.subTest(endpoint=name):
                with socket.create_connection(('127.0.0.1', port), timeout=5) as connection:
                    connection.settimeout(10)
                    payload = frame('PING|fragmentado_ñ')
                    for chunk in (payload[:1], payload[1:3], payload[3:7], payload[7:]):
                        connection.sendall(chunk)
                        time.sleep(.005)
                    self.assertTrue(receive(connection).startswith('PONG|fragmentado_ñ|'))
                    connection.sendall(frame('PING|first') + frame('PING|second'))
                    self.assertTrue(receive(connection).startswith('PONG|first|'))
                    self.assertTrue(receive(connection).startswith('PONG|second|'))

    def test_invalid_and_truncated_frames(self):
        payloads = [struct.pack('!I', size) for size in (0, MAX_BYTES+1, 0xffffffff)]
        payloads += [b'\x00\x00', struct.pack('!I', 10), struct.pack('!I', 10)+b'PI']
        for name, port in self.endpoints.items():
            for payload in payloads:
                with self.subTest(endpoint=name, payload=payload):
                    with socket.create_connection(('127.0.0.1', port), timeout=5) as connection:
                        connection.settimeout(5)
                        connection.sendall(payload)
                        connection.shutdown(socket.SHUT_WR)
                        try:
                            self.assertEqual(connection.recv(1), b'')
                        except ConnectionResetError:
                            pass
                    self.assertTrue(exchange('127.0.0.1', port, 'PING|alive').startswith('PONG|alive|'))

    def test_concurrent_clients(self):
        endpoints = list(self.endpoints.values())

        def request(index):
            request_id = 'parallel' + str(index)
            message = 'CONV|' + request_id + '|3|3|2|2|1,2,3,4,5,6,7,8,9|1,0,0,-1'
            return request_id, exchange('127.0.0.1', endpoints[index % len(endpoints)], message)

        with ThreadPoolExecutor(max_workers=12) as pool:
            for request_id, reply in pool.map(request, range(48)):
                self.result(reply, request_id, 2, 2, [-4]*4)

    def test_language_configurations(self):
        for language in ('Java', 'Python'):
            port = self.start_central('Central-only-' + language, [self.endpoints[language]])
            self.assertEqual(exchange('127.0.0.1', port, 'PING|language'), 'PONG|language|' + language)
            for message, kind in TESTS[1:]:
                check(exchange('127.0.0.1', port, message), kind)

    def test_worker_failover_and_unavailability(self):
        # Reservar sin listen evita depender de un servicio ajeno para simular la caida.
        with socket.socket() as unavailable:
            unavailable.bind(('127.0.0.1', 0))
            port = self.start_central('Central-failover',
                                      [unavailable.getsockname()[1], self.endpoints['Python']])
            for message, kind in TESTS:
                check(exchange('127.0.0.1', port, message), kind)
        worker_port = self.free_port()
        worker = self.start(['java', '-cp', str(self.artifacts), 'WorkerJava', str(worker_port)],
                            worker_port, 'Java-disconnect')
        isolated = self.start_central('Central-unavailable', [worker_port])
        worker.terminate()
        worker.wait(timeout=5)
        self.assertEqual(exchange('127.0.0.1', isolated, 'PING|offline'),
                         'ERROR|offline|no_worker_available')

    def test_existing_smoke_and_cli(self):
        commands = [[sys.executable, 'tests/smoke_test.py']
                    + ['127.0.0.1:' + str(port) for port in self.endpoints.values()],
                    ['java', '-cp', str(self.artifacts), 'ClientCli', '127.0.0.1', str(self.central)]]
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_benchmark_transport(self):
        output = self.artifacts / 'benchmark-smoke.csv'
        result = subprocess.run(['java', '-cp', str(self.artifacts), 'benchmarks.Benchmark',
                                 '127.0.0.1', str(self.central), '12', '3', '1', '3',
                                 'audit-localhost', str(output), '42'], cwd=ROOT,
                                capture_output=True, timeout=30)
        (self.artifacts / 'benchmark-smoke.log').write_bytes(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with output.open(encoding='utf-8', newline='') as source:
            rows = list(csv_module.DictReader(source))
        self.assertEqual(len(rows), 12)
        self.assertEqual({row['requestId'] for row in rows}, {'b'+str(i) for i in range(12)})
        self.assertTrue(all(row['ok'] == '1' for row in rows), rows)

    def test_python_identify_transport_only(self):
        result = subprocess.run([sys.executable, 'tests/test_local.py', '127.0.0.1',
                                 str(self.endpoints['Python']), '--seed', '42'], cwd=ROOT,
                                capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run([sys.executable, 'tests/test_local.py', '127.0.0.1',
                                 str(self.central), '--seed', '42'], cwd=ROOT,
                                capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(exchange('127.0.0.1', self.central, 'IDENTIFY|pending|0'),
                         'ERROR|pending|invalid_image_size')

    def test_local_reports_connection_failure(self):
        with socket.socket() as unavailable:
            unavailable.bind(('127.0.0.1', 0))
            result = subprocess.run([sys.executable, 'tests/test_local.py', '127.0.0.1',
                                     str(unavailable.getsockname()[1])], cwd=ROOT,
                                    capture_output=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'ERROR:', result.stderr)

    def test_z_no_unhandled_python_exceptions(self):
        log = (self.artifacts / 'Python.log').read_text(encoding='utf-8', errors='replace')
        self.assertNotIn('Traceback (most recent call last)', log)


class LocalClientTests(unittest.TestCase):
    def test_partial_receives(self):
        reply = 'RESULT|id|1|3|0.2,0.3,0.5'
        packet = frame(reply)

        class FragmentedSocket:
            def sendall(self, payload):
                self.sent = payload

            def recv(self, count):
                nonlocal packet
                data, packet = packet[:1], packet[1:]
                return data

        self.assertEqual(send_msg(FragmentedSocket(), 'PING|id'), reply)

    def test_invalid_identify_responses(self):
        replies = ['ERROR|id|failed', 'RESULT|wrong|1|3|0.2,0.3,0.5',
                   'RESULT|id|1|3|NaN,0,1', 'RESULT|id|1|3|0.2,0.3',
                   'RESULT|id|1|3|0.2,0.3,0.4', 'RESULT|id|1|3|-1,1,1']
        for reply in replies:
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                validate_identify(reply, 'id')


if __name__ == '__main__':
    unittest.main(verbosity=2)
