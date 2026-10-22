"""ETAPA 3: operaciones TCP reales en Java/Python/C#, solo biblioteca estandar.

python tests/test_distributed.py
Tolerancia absoluta 1e-8: doubles IEEE754 y diferente orden de suma entre runtimes.
Los proxies solo alteran transporte/orden para pruebas de fallo; todo resultado
aceptado se calcula en workers reales con los pesos originales.
"""
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import unittest

import test_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_python import CNN, CLASSES
from cnn_python.dataset import load_split
from cnn_python.images import read_image
from cnn_python.model import softmax


class Forwarder:
    """Proxy TCP hacia un worker real; hooks de fallos y medicion de barreras."""
    def __init__(self, destination, before=None, after=None):
        self.destination, self.before, self.after = destination, before, after
        self.events, self.clients, self.handlers, self.errors = [], [], [], []
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.port = self.listener.getsockname()[1]
        self.listener.listen(64)
        self.listener.settimeout(.1)

    def record(self, kind, fields):
        with self.lock:
            self.events.append((kind, time.perf_counter(), fields))

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
            thread = threading.Thread(target=self.handle, args=(client,), daemon=True)
            self.handlers.append(thread)
            thread.start()

    def handle(self, client):
        with client:
            client.settimeout(5)
            try:
                message = protocol.receive(client)
                fields = message.split('|')
                self.record('start', fields)
                if self.before and self.before(fields) is False:
                    return
                reply = protocol.exchange('127.0.0.1', self.destination, message)
                if self.after:
                    reply = self.after(fields, reply)
                self.record('end', fields)
                client.sendall(protocol.frame(reply))
            except (OSError, RuntimeError):
                return  # Desconexion real o inyectada: el coordinador debe detectarla.
            except Exception as error:
                with self.lock:
                    self.errors.append(error)

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


class DistributedTests(protocol.ProtocolTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = CNN.load()
        cls.examples = load_split(split='test')
        cls.distributed = cls.central_for('Distributed', list(cls.endpoints.values())[:3],
                                         traceValues='true')

    @classmethod
    def central_for(cls, name, ports, **settings):
        port = cls.free_port()
        command = ['java'] + ['-Dopenaicubes.'+key+'='+str(value)
                              for key, value in settings.items()]
        command += ['-cp', str(cls.artifacts), 'CentralServer', str(port)]
        cls.start(command + ['127.0.0.1:'+str(p) for p in ports], port, name)
        return port

    def request(self, index=0, request_id='image', mode='DISTRIBUTED', port=None):
        pixels, _ = self.examples[index]
        return protocol.exchange('127.0.0.1', port or self.distributed,
                                 'IDENTIFY|'+request_id+'|'+protocol.csv(pixels)+'|'+mode)

    def probabilities(self, reply, request_id, expected):
        self.result(reply, request_id, 1, 3, expected)
        values = [float(value) for value in reply.split('|')[4].split(',')]
        self.assertAlmostEqual(sum(values), 1, delta=1e-8)
        return values

    def prediction(self, reply, request_id, index):
        expected = self.model.predict(self.examples[index][0])
        values = self.probabilities(reply, request_id, expected.probabilities)
        self.assertEqual(max(range(3), key=values.__getitem__), expected.index)
        return values

    def part(self, reply, request_id, task_id, rows, cols, expected):
        fields = reply.split('|')
        self.assertEqual(len(fields), 7, reply)
        self.assertEqual(fields[:5], ['PART', request_id, task_id, str(rows), str(cols)])
        self.assertGreaterEqual(int(fields[6]), 0)
        values = [float(value) for value in fields[5].split(',')]
        self.assertEqual(len(values), len(expected))
        self.assertEqual(len(values), rows*cols)
        for actual, reference in zip(values, expected):
            self.assertTrue(math.isfinite(actual))
            self.assertAlmostEqual(actual, reference, delta=1e-8)
        return values

    def log(self, name, request_id, kind='TASK'):
        return [line.split('|') for line in (self.artifacts/(name+'.log'))
                .read_text(encoding='utf-8', errors='replace').splitlines()
                if line.startswith(kind+'|'+request_id+'|')]

    def test_capabilities_and_raw_jobs_all_languages(self):
        for language, threads in [('Java', 4), ('Python', 1), ('CSharp', 4)]:
            port = self.endpoints[language]
            with self.subTest(language=language):
                caps = protocol.exchange('127.0.0.1', port, 'CAPS|cap').split('|')
                self.assertEqual(caps[:3], ['CAPS', 'cap', language])
                self.assertTrue({'CONV', 'DENSE'} <= set(caps[3].split(',')))
                self.assertEqual(int(caps[4]), threads)
                conv = 'JOB|raw|conv|CONV|3|3|2|2|1,2,3,4,5,6,7,8,9|1,0,0,-1'
                self.part(protocol.exchange('127.0.0.1', port, conv),
                          'raw', 'conv', 2, 2, [-4]*4)
                dense = 'JOB|raw|dense|DENSE|2|3|-1,2|1,0,0,1,1,1|0.25,-0.5,1'
                self.part(protocol.exchange('127.0.0.1', port, dense),
                          'raw', 'dense', 1, 3, [-.75, 1.5, 2])

    def test_job_validation_consistent_and_connection_reusable(self):
        cases = [
            ('JOB', 'unknown', 'unknown', 'invalid_job_id'),
            ('JOB|req||DENSE', 'req', 'unknown', 'invalid_job_id'),
            ('JOB|bad id|task|DENSE', 'unknown', 'task', 'invalid_job_id'),
            ('JOB|req|task', 'req', 'task', 'invalid_job_fields'),
            ('JOB|req|task|OTHER', 'req', 'task', 'unsupported_operation'),
            ('JOB|req|task|CONV|1|1|2|2|1|1', 'req', 'task', 'invalid_dimensions'),
            ('JOB|req|task|DENSE|0|1|1|1|0', 'req', 'task', 'invalid_dimensions'),
            ('JOB|req|task|DENSE|1000000|2|1|1|0', 'req', 'task', 'invalid_dimensions'),
            ('JOB|req|task|DENSE|+1|1|1|1|0', 'req', 'task', 'invalid_dimensions'),
            ('JOB|req|task|DENSE|1|1|1|1', 'req', 'task', 'invalid_job_fields'),
            ('JOB|req|task|DENSE|2|1|1|1,1|0', 'req', 'task', 'invalid_numbers'),
            ('JOB|req|task|DENSE|1|1|1e308|2|0', 'req', 'task', 'nonfinite_result'),
            ('JOB|req|task|CONV|1|1|1|1|1e308|2', 'req', 'task', 'nonfinite_result'),
        ]
        for token in ('NaN', 'Infinity', '1e999', '1f', '0x1p0', ' 1', '1\n', '1_0', '\u0661'):
            cases.append(('JOB|req|task|DENSE|1|1|'+token+'|1|0',
                          'req', 'task', 'invalid_numbers'))
        for language in ('Java', 'Python', 'CSharp'):
            with socket.create_connection(('127.0.0.1', self.endpoints[language]), timeout=10) as client:
                for message, request_id, task_id, code in cases:
                    with self.subTest(language=language, message=message):
                        client.sendall(protocol.frame(message))
                        self.assertEqual(protocol.receive(client),
                                         '|'.join(['ERROR', request_id, task_id, code]))
                client.sendall(protocol.frame('PING|alive')+protocol.frame('CAPS|alive'))
                self.assertEqual(protocol.receive(client), 'PONG|alive|'+language)
                self.assertTrue(protocol.receive(client).startswith('CAPS|alive|'+language+'|'))

    def test_convolution_split_and_dense_split_between_two_workers(self):
        pixels = self.examples[117][0]
        state, weights = self.model.trace(pixels), self.model.weights
        commands = []
        for channel, language in enumerate(('Java', 'Python')):
            command = 'JOB|split|conv-'+str(channel)+'|CONV|16|16|3|3|'+protocol.csv(pixels)
            command += '|'+protocol.csv(weights['FILTER'+str(channel+1)])
            commands.append((language, command))
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda item: protocol.exchange(
                '127.0.0.1', self.endpoints[item[0]], item[1]), commands))
        for channel, reply in enumerate(replies):
            self.part(reply, 'split', 'conv-'+str(channel), 14, 14, state['convolution'][channel])
        hidden = []
        for start, language in ((0, 'CSharp'), (4, 'Java')):
            command = '|'.join(['JOB', 'split', 'hidden-'+str(start), 'DENSE', '98', '4',
                                protocol.csv(state['flat']), protocol.csv(weights['W1'][start*98:(start+4)*98]),
                                protocol.csv(weights['B1'][start:start+4])])
            hidden.append((language, command))
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda item: protocol.exchange(
                '127.0.0.1', self.endpoints[item[0]], item[1]), hidden))
        actual = []
        for start, reply in zip((0, 4), replies):
            actual += self.part(reply, 'split', 'hidden-'+str(start), 1, 4, state['prehidden'][start:start+4])
        actual = [max(0, value) for value in actual]
        logits = []
        for start, end, language in ((0, 2, 'CSharp'), (2, 3, 'Java')):
            command = '|'.join(['JOB', 'split', 'output-'+str(start), 'DENSE', '8', str(end-start),
                                protocol.csv(actual), protocol.csv(weights['W2'][start*8:end*8]),
                                protocol.csv(weights['B2'][start:end])])
            logits += self.part(protocol.exchange('127.0.0.1', self.endpoints[language], command),
                                'split', 'output-'+str(start), 1, end-start, state['logits'][start:end])
        for value, expected in zip(softmax(logits), state['probabilities']):
            self.assertAlmostEqual(value, expected, delta=1e-8)

    def test_300_images_both_modes_probabilities_and_classes(self):
        maximum, correct = 0.0, [0, 0]
        # Conexion persistente: 600 inferencias completas, mismas 300 imagenes/pesos.
        with socket.create_connection(('127.0.0.1', self.distributed), timeout=10) as client:
            client.settimeout(20)
            for index, (pixels, label) in enumerate(self.examples):
                vectors = []
                for number, mode in enumerate(('SEQUENTIAL', 'DISTRIBUTED')):
                    request_id = mode+'-'+str(index)
                    client.sendall(protocol.frame('IDENTIFY|'+request_id+'|'+protocol.csv(pixels)+'|'+mode))
                    values = self.prediction(protocol.receive(client), request_id, index)
                    vectors.append(values)
                    correct[number] += max(range(3), key=values.__getitem__) == label
                maximum = max(maximum, *(abs(a-b) for a, b in zip(*vectors)))
        self.assertEqual(correct, [300, 300])
        self.assertLessEqual(maximum, 1e-8)
        print('ETAPA3: test=300; correctos SEQUENTIAL/DISTRIBUTED='+str(correct)
              +'; max |p_seq-p_dist|='+format(maximum, '.17g'), flush=True)

    def test_intermediate_tensors_and_real_three_language_trace(self):
        request_id = 'tensor-check'
        self.prediction(self.request(233, request_id), request_id, 233)
        state = self.model.trace(self.examples[233][0])
        expected = {'conv-0': state['convolution'][0], 'conv-1': state['convolution'][1],
                    'pool-0': state['pooled'][0], 'pool-1': state['pooled'][1],
                    'flatten': state['flat'], 'hidden-linear': state['prehidden'],
                    'hidden-relu': state['hidden'], 'logits': state['logits'],
                    'probabilities': state['probabilities']}
        tensors = self.log('Distributed', request_id, 'TENSOR')
        self.assertEqual({row[3] for row in tensors}, set(expected))
        maximum = 0.0
        for row in tensors:
            values = [float(x) for x in row[6].split(',')]
            self.assertEqual(len(values), int(row[4])*int(row[5]))
            self.assertEqual(len(values), len(expected[row[3]]))
            for value, reference in zip(values, expected[row[3]]):
                maximum = max(maximum, abs(value-reference))
                self.assertAlmostEqual(value, reference, delta=1e-8)
        tasks = self.log('Distributed', request_id)
        self.assertEqual(len(tasks), 8)
        self.assertEqual(len({row[2] for row in tasks}), 1)
        self.assertEqual(len({row[3] for row in tasks}), 8)
        self.assertEqual({row[5] for row in tasks}, {'Java', 'Python', 'CSharp'})
        self.assertEqual(len({row[4] for row in tasks}), 3)
        for row in tasks:
            self.assertEqual(row[9], 'OK')
            self.assertGreaterEqual(int(row[7]), 0)
            self.assertGreater(int(row[8]), 0)
        print('ETAPA3: max error absoluto intermedio='+format(maximum, '.17g'), flush=True)
        for row in tasks:
            print('|'.join(row), flush=True)

    def test_multiple_clients_reusing_external_ids_are_isolated(self):
        def infer(number):
            index = (number*71) % len(self.examples)
            mode = 'SEQUENTIAL' if number % 4 == 0 else 'DISTRIBUTED'
            reply = self.request(index, 'shared-client-id', mode)
            self.prediction(reply, 'shared-client-id', index)
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(infer, range(48)))
        tasks = self.log('Distributed', 'shared-client-id')
        runs = {row[2] for row in tasks}
        self.assertEqual(len(runs), 36)
        for run in runs:
            self.assertEqual(len([row for row in tasks if row[2] == run]), 8)

    def test_out_of_order_concurrent_chunks_and_layer_barriers(self):
        conv_barrier, hidden_barrier = threading.Barrier(2), threading.Barrier(3)

        def before(fields):
            if fields[0] == 'JOB':
                if fields[3] == 'CONV':
                    conv_barrier.wait(timeout=5)
                elif fields[2].startswith('hidden-'):
                    hidden_barrier.wait(timeout=5)

        def after(fields, reply):
            if fields[0] == 'JOB' and fields[2] == 'conv-0':
                time.sleep(.12)
            return reply

        with Forwarder(self.endpoints['Java'], before, after) as java, \
                Forwarder(self.endpoints['Python'], before, after) as python, \
                Forwarder(self.endpoints['CSharp'], before, after) as csharp:
            port = self.central_for('OutOfOrder', [java.port, python.port, csharp.port])
            self.prediction(self.request(154, 'reordered', port=port), 'reordered', 154)
            events = sorted(java.events+python.events+csharp.events, key=lambda event: event[1])
            self.assertFalse(java.errors+python.errors+csharp.errors)
            jobs = [event for event in events if event[2][0] == 'JOB']
            self.assertEqual(len([event for event in jobs if event[0] == 'start']), 8)
            conv_end = [event for event in jobs if event[0] == 'end' and event[2][3] == 'CONV']
            self.assertEqual([event[2][2] for event in conv_end], ['conv-1', 'conv-0'])
            hidden_start = [event[1] for event in jobs if event[0] == 'start' and event[2][2].startswith('hidden-')]
            hidden_end = [event[1] for event in jobs if event[0] == 'end' and event[2][2].startswith('hidden-')]
            output_start = [event[1] for event in jobs if event[0] == 'start' and event[2][2].startswith('output-')]
            self.assertGreaterEqual(min(hidden_start), max(event[1] for event in conv_end))
            self.assertGreaterEqual(min(output_start), max(hidden_end))
            self.assertTrue(all(event[2][0] in ('PING', 'CAPS', 'JOB') for event in events))

    def test_worker_dies_during_inference_and_rejoins(self):
        worker_port = self.free_port()
        command = ['java', '-cp', str(self.artifacts), 'WorkerJava', str(worker_port)]
        worker = self.start(command, worker_port, 'Java-killed')
        killed = threading.Event()

        def before(fields):
            if fields[0] == 'JOB' and not killed.is_set():
                killed.set()
                worker.terminate()
                worker.wait(timeout=5)
                return False

        with Forwarder(worker_port, before) as proxy:
            port = self.central_for('WorkerRecovery',
                                    [proxy.port, self.endpoints['Python'], self.endpoints['CSharp']])
            self.prediction(self.request(41, 'worker-killed', port=port), 'worker-killed', 41)
            self.assertTrue(killed.is_set())
            trace = self.log('WorkerRecovery', 'worker-killed')
            self.assertTrue(any(row[9].startswith('FAILED:') for row in trace))
            self.assertEqual({row[5] for row in trace if row[9] == 'OK'}, {'Python', 'CSharp'})
            self.start(command, worker_port, 'Java-restarted')
            self.prediction(self.request(202, 'worker-back', port=port), 'worker-back', 202)
            self.assertEqual({row[5] for row in self.log('WorkerRecovery', 'worker-back')},
                             {'Java', 'Python', 'CSharp'})

    def test_invalid_partial_result_retried_and_never_accepted(self):
        corruptions = [lambda p: p.__setitem__(1, 'other-request'),
                       lambda p: p.__setitem__(2, 'other-task'),
                       lambda p: p.__setitem__(3, '1'),
                       lambda p: p.__setitem__(5, 'NaN'),
                       lambda p: p.__setitem__(6, '-1')]
        for number, corrupt in enumerate(corruptions):
            def after(fields, reply):
                if fields[0] == 'JOB':
                    parts = reply.split('|')
                    corrupt(parts)
                    return '|'.join(parts)
                return reply
            with Forwarder(self.endpoints['Java'], after=after) as proxy:
                name = 'BadPart'+str(number)
                port = self.central_for(name, [proxy.port, self.endpoints['Python'], self.endpoints['CSharp']])
                request_id = 'bad-part-'+str(number)
                self.prediction(self.request(number, request_id, port=port), request_id, number)
                rows = self.log(name, request_id)
                self.assertTrue(any(row[9] == 'FAILED:invalid_partial_result' for row in rows))
                self.assertFalse(any(row[5] == 'Java' and row[9] == 'OK' for row in rows))

    def test_timeout_is_bounded_and_next_request_recovers(self):
        stall = threading.Event()
        stall.set()

        def before(fields):
            if fields[0] == 'JOB' and stall.is_set():
                time.sleep(.6)
                return False

        with Forwarder(self.endpoints['Java'], before) as java, \
                Forwarder(self.endpoints['Python'], before) as python:
            port = self.central_for('BoundedTimeout', [java.port, python.port],
                                    workerTimeoutMs=120, identifyTimeoutMs=350)
            start = time.monotonic()
            reply = self.request(12, 'bounded-timeout', port=port)
            self.assertIn(reply, ['ERROR|bounded-timeout|worker_timeout',
                                  'ERROR|bounded-timeout|distributed_timeout'])
            self.assertLess(time.monotonic()-start, 3)
            stall.clear()
            self.prediction(self.request(12, 'after-timeout', port=port), 'after-timeout', 12)

    def test_one_node_cannot_silently_fall_back_to_whole_identify(self):
        port = self.central_for('OneNode', [self.endpoints['Python']])
        self.assertEqual(self.request(0, 'need-two', port=port), 'ERROR|need-two|insufficient_workers')
        self.prediction(self.request(0, 'seq-one', 'SEQUENTIAL', port), 'seq-one', 0)

    def test_all_partials_invalid_produce_error_and_legacy_nodes_are_skipped(self):
        def after(fields, reply):
            if fields[0] == 'JOB':
                parts = reply.split('|')
                parts[1] = 'unrelated-request'
                return '|'.join(parts)
            return reply

        with Forwarder(self.endpoints['Java'], after=after) as java, \
                Forwarder(self.endpoints['Python'], after=after) as python:
            port = self.central_for('AllBadParts', [java.port, python.port])
            self.assertEqual(self.request(0, 'all-bad', port=port),
                             'ERROR|all-bad|invalid_partial_result')

        def no_caps(fields, reply):
            if fields[0] == 'CAPS':
                return 'ERROR|'+fields[1]+'|unsupported_command'
            return reply

        with Forwarder(self.endpoints['Java'], after=no_caps) as legacy:
            port = self.central_for('LegacyCaps',
                                    [legacy.port, self.endpoints['Python'], self.endpoints['CSharp']])
            self.prediction(self.request(203, 'skip-legacy', port=port), 'skip-legacy', 203)
            self.assertFalse(any(fields[0] == 'JOB' for _, _, fields in legacy.events))

    def test_invalid_inputs_and_mode_preserve_stable_errors(self):
        pixels = protocol.csv(self.examples[0][0])
        cases = [('IDENTIFY|bad|0|DISTRIBUTED', 'ERROR|bad|invalid_image_size'),
                 ('IDENTIFY|bad|'+pixels+'|other', 'ERROR|bad|invalid_identify_fields'),
                 ('IDENTIFY|bad id|'+pixels+'|DISTRIBUTED', 'ERROR|unknown|invalid_request_id')]
        for token in ('NaN', '-0.1', '1.1', '1e999', '1f'):
            cases.append(('IDENTIFY|bad|'+','.join([token]+['0']*255)+'|DISTRIBUTED',
                          'ERROR|bad|invalid_pixels'))
        for message, expected in cases:
            self.assertEqual(protocol.exchange('127.0.0.1', self.distributed, message), expected)
        self.prediction(self.request(0, 'valid-after-errors'), 'valid-after-errors', 0)

    def test_configurable_threads_chunks_and_missing_model(self):
        ports = [self.endpoints[language] for language in ('Java', 'Python', 'CSharp')]
        port = self.central_for('Configured', ports, taskThreads=1, denseChunkSize=8)
        self.prediction(self.request(201, 'configured', port=port), 'configured', 201)
        tasks = self.log('Configured', 'configured')
        self.assertEqual(len(tasks), 6)  # 2 CONV + 2 hidden + 2 output, incluso chunk=8.
        port = self.central_for('MissingModel', ports, weights=str(self.artifacts/'absent-weights.txt'))
        self.assertEqual(self.request(0, 'no-model', port=port), 'ERROR|no-model|model_unavailable')

    def test_java_cli_both_modes_three_classes(self):
        for mode in ('SEQUENTIAL', 'DISTRIBUTED'):
            for label, name in enumerate(CLASSES):
                image = ROOT/'model/examples'/(name+'-0.png')
                request_id = mode+'-'+name
                result = subprocess.run(['java', '-cp', str(self.artifacts), 'ClientCli',
                                         '127.0.0.1', str(self.distributed), 'IDENTIFY', str(image),
                                         request_id, mode], cwd=ROOT, capture_output=True, text=True,
                                        encoding='utf-8', errors='replace', timeout=25)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                reply = next(line for line in result.stdout.splitlines() if line.startswith('RESULT|'))
                values = self.probabilities(reply, request_id, self.model.predict(read_image(image)).probabilities)
                self.assertEqual(max(range(3), key=values.__getitem__), label)
                self.assertIn('Modo: '+mode, result.stdout)


def load_tests(loader, tests, pattern):
    # Hereda utilidades/fixture; las pruebas legacy se ejecutan en test_protocol.
    return unittest.TestSuite(DistributedTests(name) for name in sorted(DistributedTests.__dict__)
                              if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
