"""ETAPA 2: integracion real Java -> Java -> Python -> CNN, sin dependencias.

python tests/test_identify.py
Los peers de fallo solo inyectan errores de transporte; la precision se mide
exclusivamente contra el worker real y los 300 ejemplos test de la ETAPA 1.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock
import zipfile
import zlib

import test_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_python import CNN, CLASSES
from cnn_python.dataset import DEFAULT_DATASET, load_split
from cnn_python.images import PNG, encode_png, read_image

STALL = object()


class Drip:
    def __init__(self, data):
        self.data = data


class FaultPeer:
    """Peer TCP acotado para comprobar errores y capturar mensajes del cliente."""
    def __init__(self, response=None, language='Python'):
        self.response = response
        self.language = language
        self.requests = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.port = self.listener.getsockname()[1]
        self.listener.listen(32)
        self.listener.settimeout(.1)
        self.handlers = []
        self.clients = []

    def __enter__(self):
        self.thread = threading.Thread(target=self.accept, daemon=True)
        self.thread.start()
        return self

    def accept(self):
        while not self.stop.is_set():
            try:
                client, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self.clients.append(client)
            handler = threading.Thread(target=self.handle, args=(client,), daemon=True)
            self.handlers.append(handler)
            handler.start()

    def handle(self, client):
        with client:
            client.settimeout(3)
            try:
                while not self.stop.is_set():
                    message = protocol.receive(client)
                    with self.lock:
                        self.requests.append(message)
                    fields = message.split('|')
                    if fields[0] == 'PING':
                        response = 'PONG|' + fields[1] + '|' + self.language
                    else:
                        response = self.response(fields) if self.response else 'ERROR|' + fields[1] + '|unsupported_command'
                    if response is STALL:
                        self.stop.wait(5)
                        return
                    if isinstance(response, Drip):
                        for value in response.data:
                            client.sendall(bytes([value]))
                            if self.stop.wait(.04):
                                return
                    elif isinstance(response, bytes):
                        client.sendall(response)
                        return
                    else:
                        client.sendall(protocol.frame(response))
            except (OSError, RuntimeError, ValueError, AssertionError):
                return

    def __exit__(self, *args):
        self.stop.set()
        self.listener.close()
        self.thread.join(timeout=2)
        for client in self.clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()
        for handler in self.handlers:
            handler.join(timeout=2)


class IdentifyTests(protocol.ProtocolTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = CNN.load()
        cls.examples = load_split(split='test')

    def request(self, index=0, request_id='image', port=None):
        pixels, _ = self.examples[index]
        return protocol.exchange('127.0.0.1', port or self.central,
                                 'IDENTIFY|' + request_id + '|' + protocol.csv(pixels))

    def prediction(self, reply, request_id, pixels):
        expected = self.model.predict(pixels)
        self.result(reply, request_id, 1, 3, expected.probabilities)
        values = [float(value) for value in reply.split('|')[4].split(',')]
        self.assertAlmostEqual(sum(values), 1, delta=1e-8)
        return max(range(3), key=values.__getitem__)

    def central_for(self, name, ports, timeout_ms=None):
        if timeout_ms is None:
            return self.start_central(name, ports)
        port = self.free_port()
        command = ['java', '-Dopenaicubes.workerTimeoutMs='+str(timeout_ms),
                   '-Dopenaicubes.identifyTimeoutMs='+str(timeout_ms*4),
                   '-Dopenaicubes.clientTimeoutMs='+str(max(timeout_ms*6, 2000)),
                   '-cp', str(self.artifacts), 'CentralServer', str(port)]
        self.start(command+['127.0.0.1:'+str(p) for p in ports], port, name)
        return port

    def java_client(self, path, request_id='cli-image', port=None, timeout_ms=None):
        command = ['java']
        if timeout_ms is not None:
            command += ['-Dopenaicubes.clientTimeoutMs='+str(timeout_ms)]
        command += ['-cp', str(self.artifacts), 'ClientCli', '127.0.0.1',
                    str(port or self.central), 'IDENTIFY', str(path)]
        if request_id is not None:
            command.append(request_id)
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=20)

    def test_identify_300_independent_images_match_local(self):
        confusion = [[0]*3 for _ in range(3)]
        with socket.create_connection(('127.0.0.1', self.central), timeout=5) as connection:
            connection.settimeout(15)
            for index, (pixels, label) in enumerate(self.examples):
                request_id = 'heldout-'+str(index)
                connection.sendall(protocol.frame('IDENTIFY|'+request_id+'|'+protocol.csv(pixels)))
                predicted = self.prediction(protocol.receive(connection), request_id, pixels)
                confusion[label][predicted] += 1
        self.assertEqual(confusion, [[100, 0, 0], [0, 100, 0], [0, 0, 100]])
        (self.artifacts/'identify-accuracy.json').write_text(json.dumps(
            {'samples': 300, 'accuracy': 1.0, 'confusion': confusion, 'absolute_tolerance': 1e-8}, indent=2))

    def test_identify_java_cli_three_classes(self):
        for label, name in enumerate(CLASSES):
            image = ROOT/'model/examples'/(name+'-0.png')
            result = self.java_client(image, 'java-'+name)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            reply = next(line for line in result.stdout.splitlines() if line.startswith('RESULT|'))
            self.assertEqual(self.prediction(reply, 'java-'+name, read_image(image)), label)
            self.assertIn('Clase: '+name, result.stdout)
            self.assertIn('Tiempo de respuesta:', result.stdout)
        result = self.java_client(ROOT/'model/examples/cruz-0.png', request_id=None)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('Clase: cruz', result.stdout)

    def test_identify_java_preprocessing_matches_every_pixel(self):
        images = []
        gray = self.artifacts/'gray-no-class.bin'
        gray.write_bytes(encode_png(23, 19, [(i*17) % 256 for i in range(23*19)]))
        images.append(gray)
        for magic in ('P2', 'P5'):
            path = self.artifacts/(magic+'.pgm')
            samples = [10, 32, 50, 100, 0, 25]
            body = bytes(samples) if magic == 'P5' else ' '.join(map(str, samples)).encode()
            path.write_bytes((magic+'\r\n# test\r\n3 2\r\n100\r\n').encode()+body)
            images.append(path)

        def chunk(kind, body):
            return struct.pack('!I', len(body))+kind+body+struct.pack('!I', zlib.crc32(kind+body))

        rgba = self.artifacts/'rgba.png'
        header = struct.pack('!IIBBBBB', 2, 2, 8, 6, 0, 0, 0)
        raw = bytes([0, 255, 0, 0, 128, 0, 255, 0, 255,
                     0, 0, 0, 255, 0, 255, 255, 255, 200])
        rgba.write_bytes(PNG+chunk(b'IHDR', header)+chunk(b'IDAT', zlib.compress(raw))+chunk(b'IEND', b''))
        images.append(rgba)
        with zipfile.ZipFile(DEFAULT_DATASET) as archive:
            path = self.artifacts/'heldout.pgm'
            path.write_bytes(archive.read('test/000000.pgm'))
            images.append(path)

        # Captura de transporte del cliente; la respuesta usa la CNN real sobre los pixeles recibidos.
        def infer(fields):
            result = self.model.predict([float(value) for value in fields[2].split(',')])
            return 'RESULT|'+fields[1]+'|1|3|'+protocol.csv(result.probabilities)

        with FaultPeer(infer) as peer:
            for index, path in enumerate(images):
                result = self.java_client(path, 'pixels-'+str(index), peer.port)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                expected = read_image(path)
                actual = [float(value) for value in peer.requests[-1].split('|')[2].split(',')]
                self.assertEqual(len(actual), 256)
                for x, y in zip(actual, expected):
                    self.assertAlmostEqual(x, y, delta=1e-12)

    def test_identify_invalid_input_and_persistent_connection(self):
        pixels = protocol.csv(self.examples[0][0])
        cases = [('IDENTIFY', 'unknown', 'invalid_request_id'),
                 ('IDENTIFY||'+pixels, 'unknown', 'invalid_request_id'),
                 ('IDENTIFY|bad id|'+pixels, 'unknown', 'invalid_request_id'),
                 ('IDENTIFY|bad\n|'+pixels, 'unknown', 'invalid_request_id'),
                 ('IDENTIFY|'+'x'*129+'|'+pixels, 'unknown', 'invalid_request_id'),
                 ('IDENTIFY|shape', 'shape', 'invalid_identify_fields'),
                 ('IDENTIFY|shape|'+pixels+'|extra', 'shape', 'invalid_identify_fields'),
                 ('IDENTIFY|size|', 'size', 'invalid_image_size'),
                 ('IDENTIFY|size|'+','.join(['0']*255), 'size', 'invalid_image_size'),
                 ('IDENTIFY|size|'+','.join(['0']*257), 'size', 'invalid_image_size')]
        for bad in ('NaN', 'Infinity', '1e999', '-0.1', '1.001', '', 'abc', '0x0p0',
                    '0.0f', ' 0.1', '0_0', '0'*65):
            cases.append(('IDENTIFY|pixels|'+','.join([bad]+['0']*255), 'pixels', 'invalid_pixels'))
        for port in (self.central, self.endpoints['Python']):
            with socket.create_connection(('127.0.0.1', port), timeout=5) as connection:
                connection.settimeout(10)
                for message, request_id, reason in cases:
                    with self.subTest(port=port, reason=reason, message=message[:40]):
                        connection.sendall(protocol.frame(message))
                        self.assertEqual(protocol.receive(connection), 'ERROR|'+request_id+'|'+reason)
                request_id = 'imagen_ñ_☑'
                connection.sendall(protocol.frame('IDENTIFY|'+request_id+'|'+pixels))
                self.prediction(protocol.receive(connection), request_id, self.examples[0][0])
                connection.sendall(protocol.frame('PING|recovered'))
                self.assertTrue(protocol.receive(connection).startswith('PONG|recovered|'))

    def test_identify_concurrent_unique_and_repeated_ids(self):
        def identify(index):
            sample = (index % 3)*100+(index % 10)
            request_id = 'concurrent-'+str(index) if index < 36 else 'shared-id'
            return sample, request_id, self.request(sample, request_id)

        with ThreadPoolExecutor(max_workers=12) as pool:
            for sample, request_id, reply in pool.map(identify, range(60)):
                pixels, label = self.examples[sample]
                self.assertEqual(self.prediction(reply, request_id, pixels), label)

    def test_identify_more_than_sixteen_persistent_clients(self):
        connections = []
        try:
            for index in range(24):
                connection = socket.create_connection(('127.0.0.1', self.central), timeout=5)
                connection.settimeout(5)
                connections.append(connection)
                pixels = self.examples[(index % 3)*100][0]
                connection.sendall(protocol.frame('IDENTIFY|persistent-'+str(index)+'|'+protocol.csv(pixels)))
            for index, connection in enumerate(connections):
                pixels = self.examples[(index % 3)*100][0]
                self.prediction(protocol.receive(connection), 'persistent-'+str(index), pixels)
                connection.sendall(protocol.frame('PING|keepalive'))
                self.assertTrue(protocol.receive(connection).startswith('PONG|keepalive|'))
        finally:
            for connection in connections:
                connection.close()

    def test_identify_mixed_fragmented_and_coalesced_messages(self):
        request_id, pixels = 'fragmented', self.examples[100][0]
        packet = protocol.frame('IDENTIFY|'+request_id+'|'+protocol.csv(pixels))
        with socket.create_connection(('127.0.0.1', self.central), timeout=5) as connection:
            connection.settimeout(10)
            for part in (packet[:1], packet[1:3], packet[3:9], packet[9:]):
                connection.sendall(part)
                time.sleep(.002)
            self.prediction(protocol.receive(connection), request_id, pixels)
            conv, kind = protocol.TESTS[1]
            connection.sendall(protocol.frame(conv)+protocol.frame('IDENTIFY|second|'+protocol.csv(pixels)))
            protocol.check(protocol.receive(connection), kind)
            self.prediction(protocol.receive(connection), 'second', pixels)

    def test_identify_never_sent_to_java_or_csharp(self):
        with FaultPeer(language='Java') as java, FaultPeer(language='CSharp') as csharp:
            port = self.central_for('Identify-routing', [java.port, csharp.port, self.endpoints['Python']])
            for index in (0, 100, 200):
                self.prediction(self.request(index, 'routing', port), 'routing', self.examples[index][0])
            for peer in (java, csharp):
                self.assertTrue(peer.requests)
                self.assertTrue(all(message.startswith('PING|') for message in peer.requests), peer.requests)

    def test_identify_without_python_worker(self):
        port = self.central_for('Identify-no-python', [self.endpoints['Java'], self.endpoints['CSharp']])
        self.assertEqual(self.request(0, 'offline', port), 'ERROR|offline|no_identify_worker_available')

    def test_identify_worker_disconnect_and_restart(self):
        port = self.free_port()
        worker = self.start([sys.executable, 'workers/python/worker.py', str(port)], port, 'Python-isolated')
        central = self.central_for('Identify-disconnect', [self.endpoints['Java'], port, self.endpoints['CSharp']])
        self.prediction(self.request(0, 'before', central), 'before', self.examples[0][0])
        worker.terminate()
        worker.wait(timeout=5)
        self.assertEqual(self.request(0, 'after', central), 'ERROR|after|no_identify_worker_available')
        self.start([sys.executable, 'workers/python/worker.py', str(port)], port, 'Python-restarted')
        self.prediction(self.request(200, 'restarted', central), 'restarted', self.examples[200][0])

    def test_identify_rejects_corrupt_worker_responses(self):
        responses = [lambda p: 'RESULT|wrong|1|3|0.2,0.3,0.5',
                     lambda p: 'RESULT|'+p[1]+'|2|3|0.2,0.3,0.5',
                     lambda p: 'RESULT|'+p[1]+'|1|3|NaN,0,1',
                     lambda p: 'RESULT|'+p[1]+'|1|3|0.2,0.8',
                     lambda p: 'RESULT|'+p[1]+'|1|3|0.2,0.3,0.4',
                     lambda p: 'RESULT|'+p[1]+'|1|3|-1,1,1',
                     lambda p: 'RESULT|'+p[1]+'|1|3|0.2,0.3,0.5|extra',
                     lambda p: 'ERROR|wrong|model_unavailable']
        for index, response in enumerate(responses):
            with self.subTest(case=index), FaultPeer(response) as peer:
                port = self.central_for('Identify-corrupt-'+str(index), [peer.port])
                self.assertEqual(self.request(0, 'corrupt', port), 'ERROR|corrupt|invalid_worker_response')

    def test_identify_fallback_after_corrupt_or_truncated_response(self):
        responses = [lambda p: 'RESULT|wrong|1|3|0.2,0.3,0.5',
                     lambda p: struct.pack('!I', 100)+b'RESULT|',
                     lambda p: 'ERROR|'+p[1]+'|unsupported_command']
        for index, response in enumerate(responses):
            with FaultPeer(response) as peer:
                port = self.central_for('Identify-fallback-'+str(index), [peer.port, self.endpoints['Python']])
                self.prediction(self.request(0, 'fallback', port), 'fallback', self.examples[0][0])

    def test_identify_worker_error_propagated(self):
        with FaultPeer(lambda p: 'ERROR|'+p[1]+'|model_unavailable') as peer:
            port = self.central_for('Identify-model-error', [peer.port])
            self.assertEqual(self.request(0, 'model', port), 'ERROR|model|model_unavailable')
        spec = importlib.util.spec_from_file_location('worker_errors', ROOT/'workers/python/worker.py')
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        message = 'IDENTIFY|model|'+protocol.csv(self.examples[0][0])
        with mock.patch.object(worker, 'load_weights', side_effect=FileNotFoundError('missing')):
            self.assertEqual(worker.evaluate(message), 'ERROR|model|model_unavailable')
        with mock.patch.object(worker, 'load_weights') as loader:
            loader.return_value.predict.side_effect = ValueError('calculation')
            self.assertEqual(worker.evaluate(message), 'ERROR|model|cnn_processing_error')

    def test_identify_timeout_and_slow_frame_are_bounded(self):
        modes = [lambda p: STALL,
                 lambda p: Drip(protocol.frame('RESULT|'+p[1]+'|1|3|0.2,0.3,0.5'))]
        for index, response in enumerate(modes):
            with FaultPeer(response) as peer:
                port = self.central_for('Identify-timeout-'+str(index), [peer.port], timeout_ms=250)
                started = time.monotonic()
                self.assertEqual(self.request(0, 'timeout', port), 'ERROR|timeout|identify_worker_timeout')
                self.assertLess(time.monotonic()-started, 2)
                self.assertEqual(protocol.exchange('127.0.0.1', port, 'PING|alive'), 'PONG|alive|Python')

    def test_identify_java_client_rejects_invalid_server_replies(self):
        responses = [lambda p: 'RESULT|wrong|1|3|0.2,0.3,0.5',
                     lambda p: 'RESULT|'+p[1]+'|1|3|NaN,0,1',
                     lambda p: 'RESULT|'+p[1]+'|1|3|0.2,0.3,0.4',
                     lambda p: 'ERROR|'+p[1]+'|no_identify_worker_available',
                     lambda p: struct.pack('!I', 0),
                     lambda p: struct.pack('!I', 2_000_001),
                     lambda p: struct.pack('!I', 100)+b'RESULT|',
                     lambda p: struct.pack('!I', 1)+b'\xff',
                     lambda p: STALL]
        for index, response in enumerate(responses):
            with self.subTest(case=index), FaultPeer(response) as peer:
                result = self.java_client(ROOT/'model/examples/circulo-0.png', port=peer.port, timeout_ms=400)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('ERROR:', result.stderr)
                self.assertNotIn('Clase:', result.stdout)

    def test_identify_java_client_rejects_bad_images_and_ids(self):
        for index, body in enumerate((b'bad', b'P5\n2 2\n255\nabc', b'P2\n1 1\n255\n256',
                                      b'P5\n2147483647 2147483647\n255\n')):
            path = self.artifacts/('invalid-'+str(index)+'.dat')
            path.write_bytes(body)
            result = self.java_client(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('ERROR:', result.stderr)
        for request_id in ('', 'a b', 'x'*129, 'a|b'):
            result = self.java_client(ROOT/'model/examples/circulo-0.png', request_id)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('invalid_request_id', result.stderr)

    def test_identify_invalid_utf8_and_truncation_do_not_stop_services(self):
        for port in (self.central, self.endpoints['Python']):
            for packet in (protocol.frame('IDENTIFY|partial|')[:-2], struct.pack('!I', 1)+b'\xff'):
                with socket.create_connection(('127.0.0.1', port), timeout=5) as connection:
                    connection.settimeout(5)
                    connection.sendall(packet)
                    connection.shutdown(socket.SHUT_WR)
                    try:
                        self.assertEqual(connection.recv(1), b'')
                    except ConnectionResetError:
                        pass
                self.assertTrue(protocol.exchange('127.0.0.1', port, 'PING|alive').startswith('PONG|alive|'))
        self.prediction(self.request(0, 'survived'), 'survived', self.examples[0][0])

    def test_identify_python_frame_has_total_deadline(self):
        spec = importlib.util.spec_from_file_location('worker_deadline', ROOT/'workers/python/worker.py')
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        stop = threading.Event()
        receiver, sender = socket.socketpair()
        with receiver, sender:
            sender.sendall(struct.pack('!I', 100))

            def drip():
                while not stop.wait(.03):
                    try:
                        sender.sendall(b'0')
                    except OSError:
                        return

            thread = threading.Thread(target=drip, daemon=True)
            thread.start()
            try:
                with mock.patch.object(worker, 'FRAME_TIMEOUT_SECONDS', .15):
                    started = time.monotonic()
                    with self.assertRaises(TimeoutError):
                        worker.recv_frame(receiver)
                    self.assertLess(time.monotonic()-started, .8)
            finally:
                stop.set()
                thread.join(timeout=1)


def load_tests(loader, tests, pattern):
    # Reutilizar el fixture no duplica las pruebas heredadas de la ETAPA 0.
    return unittest.TestSuite(IdentifyTests(name) for name in sorted(IdentifyTests.__dict__)
                              if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
