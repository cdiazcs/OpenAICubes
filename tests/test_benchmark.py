import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.util
import math
from pathlib import Path
import socketserver
import subprocess
import sys
import threading
import unittest

import test_protocol as protocol
from test_identify import FaultPeer, STALL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'benchmarks'))
from cnn_python import CNN
from cnn_python.dataset import load_split
import node_agent
import resource_probe
import GraficasBenchmark as graphs


class BenchmarkTests(protocol.ProtocolTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.compile(['javac', '--release', '8', '-encoding', 'UTF-8', '-cp', str(cls.artifacts),
                     '-d', str(cls.artifacts), 'benchmarks/LocalCnn.java', 'tests/BenchmarkChecks.java'], 'benchmark-checks')

    def benchmark(self, name, scenario='LOCAL_SEQUENTIAL', **values):
        output = self.artifacts/(name+'.csv')
        config = self.artifacts/(name+'.properties')
        settings = dict(scenario=scenario, images='model/examples', requests=40, clients=1,
                        warmup=10, cores=1, threads=1, **{'delay.min.ms': 0, 'delay.max.ms': 0})
        settings.update(values)
        settings['output'] = output.as_posix()
        config.write_text(''.join(str(key)+'='+str(value)+'\n' for key, value in settings.items()), encoding='utf-8')
        result = subprocess.run(['java', '-cp', str(self.artifacts), 'benchmarks.Benchmark', '--config', str(config)],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
        (self.artifacts/(name+'.benchmark.log')).write_text(result.stdout+result.stderr, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return output, graphs.properties(Path(str(output)+'.summary.properties'))

    def test_local_cnn_300_images_match_independent_python_and_parallel(self):
        examples = load_split(split='test')
        data = ''.join(','.join(map(str, pixels))+'\n' for pixels, _ in examples)
        result = subprocess.run(['java', '-cp', str(self.artifacts), 'BenchmarkChecks', 'model/weights/weights.txt'],
                                input=data, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = [[float(value) for value in line.split(',')] for line in result.stdout.splitlines()]
        self.assertEqual(len(outputs), 300)
        model, correct, maximum = CNN.load(), 0, 0
        for (pixels, label), actual in zip(examples, outputs):
            expected = model.predict(pixels)
            maximum = max(maximum, max(abs(x-y) for x, y in zip(actual, expected.probabilities)))
            correct += max(range(3), key=actual.__getitem__) == label
        self.assertEqual(correct, 300)
        self.assertLessEqual(maximum, 1e-12)
        print('ETAPA6: 300/300 Java local secuencial/paralelo; max error Python=', maximum, result.stderr.strip(), flush=True)

    def test_local_benchmark_concurrent_clients_metrics_reproducibility_and_graphs(self):
        first, a = self.benchmark('measure-sequential', clients=4, requests=120)
        second, b = self.benchmark('measure-parallel', scenario='LOCAL_PARALLEL', clients=4, requests=120,
                                   threads=2, cores=2, baseline=first.as_posix()+'.summary.properties')
        self.assertEqual(a['load.sha256'], b['load.sha256'])
        for path, summary in ((first, a), (second, b)):
            self.assertEqual(summary['errors'], '0')
            self.assertEqual(summary['configuration_valid'], 'true')
            self.assertEqual(summary['verification'], 'local_affinity_verified')
            self.assertEqual(summary['helper_processes_during_measurement'], '0')
            self.assertIsNotNone(graphs.measured(path))
            self.assertGreater(float(summary['throughput.rps']), 0)
        self.assertEqual(a['compute.peak.tasks'], '1')
        self.assertNotIn(',', a['compute.thread.ids'])
        self.assertIn(',', b['compute.thread.ids'])
        self.assertLessEqual(int(b['compute.peak.tasks']), 2)
        self.assertAlmostEqual(float(b['speedup.total']), float(a['total.s'])/float(b['total.s']), delta=1e-12)
        _, different = self.benchmark('different-load', scenario='LOCAL_PARALLEL', clients=4, requests=20,
                                      threads=2, cores=2, baseline=first.as_posix()+'.summary.properties')
        self.assertEqual(different['speedup.total'], '')
        with first.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        with second.open(newline='', encoding='utf-8') as stream:
            other = list(csv.DictReader(stream))
        self.assertEqual([(r['imagen_sha256'], r['retraso_ms'], r['cliente']) for r in rows],
                         [(r['imagen_sha256'], r['retraso_ms'], r['cliente']) for r in other])
        destination = self.artifacts/'real-graphs'
        data = graphs.generate(self.artifacts, destination)
        self.assertGreaterEqual(len(data), 2)
        import xml.etree.ElementTree as xml
        for path in destination.rglob('*.svg'):
            xml.parse(path)
        self.assertTrue(list(destination.rglob('speedup.svg')))
        damaged = self.artifacts/'damaged.csv'
        damaged.write_bytes(first.read_bytes()+b'broken')
        Path(str(damaged)+'.summary.properties').write_bytes(Path(str(first)+'.summary.properties').read_bytes())
        with self.assertRaisesRegex(ValueError, 'checksum'):
            graphs.measured(damaged)
        damaged.unlink(); Path(str(damaged)+'.summary.properties').unlink()

    def test_post_operations_all_three_languages(self):
        for language in ('Java', 'Python', 'CSharp'):
            for op, rows, cols, values, expected in [
                ('POOL', 4, 4, list(range(-8, 8)), [0, 0, 5, 7]),
                ('RELU', 1, 4, [-2, 0, .5, 4], [0, 0, .5, 4]),
                ('SOFTMAX', 1, 3, [1000, 1000, 1000], [1/3]*3),
            ]:
                with self.subTest(language=language, operation=op):
                    message = 'JOB|post-test|task|POST|'+op+'|'+str(rows)+'|'+str(cols)+'|'+protocol.csv(values)
                    reply = protocol.exchange('127.0.0.1', self.endpoints[language], message).split('|')
                    self.assertEqual(reply[:3], ['PART', 'post-test', 'task'])
                    actual = list(map(float, reply[5].split(',')))
                    self.assertEqual(len(actual), len(expected))
                    for x, y in zip(actual, expected):
                        self.assertAlmostEqual(x, y, delta=1e-12)
            bad = protocol.exchange('127.0.0.1', self.endpoints[language], 'JOB|bad|task|POST|SOFTMAX|2|1|0,1')
            self.assertTrue(bad.startswith('ERROR|bad|'), bad)

    def test_invalid_local_resource_configurations_are_rejected(self):
        for scenario, cores, threads in [('LOCAL_SEQUENTIAL', 2, 2), ('LOCAL_PARALLEL', 1, 2)]:
            result = subprocess.run(['java', '-cp', str(self.artifacts), 'benchmarks.Benchmark', '--config',
                                     'benchmarks/configs/local-sequential.properties', 'scenario='+scenario,
                                     'cores='+str(cores), 'threads='+str(threads)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires_', result.stderr)

    def test_error_causes_ids_deadline_and_failed_requests_not_successes(self):
        cases = [
            ('processing', lambda req: 'ERROR|'+req[1]+'|worker_busy'),
            ('validation', lambda req: 'RESULT|wrong-id|1|3|0.2,0.3,0.5'),
            ('numeric', lambda req: 'RESULT|'+req[1]+'|1|3|0.2,0.3,0.5'),
            ('connection_lost', lambda req: b''),
            ('timeout', lambda req: STALL),
        ]
        for cause, response in cases:
            with self.subTest(cause=cause), FaultPeer(response) as peer:
                _, summary = self.benchmark('fault-'+cause, scenario='SOCKET_CONTROL', requests=4, warmup=0,
                                            host='127.0.0.1', port=peer.port, **{'timeout.ms': 150})
                self.assertEqual(summary['success'], '0')
                self.assertEqual(summary['errors.'+cause], '4')
                self.assertEqual(summary['throughput.rps'], '0.0')
                self.assertEqual(summary['speedup.total'], '')
        _, summary = self.benchmark('fault-refused', scenario='SOCKET_CONTROL', requests=4, warmup=0,
                                    host='127.0.0.1', port=self.free_port())
        self.assertEqual(summary['errors.connection_not_sent'], '4')

    def test_distributed_profiles_nodes_limits_and_localhost_is_not_three_machines(self):
        for scenario, cores in [('DISTRIBUTED', 1), ('PARALLEL_DISTRIBUTED', 2)]:
            nodes, servers = [], []
            worker_ports = [self.free_port() for _ in range(3)]
            if cores == 2:
                while worker_ports[2]+1 in self.ports:
                    worker_ports[2] = self.free_port()
                with __import__('socket').socket() as probe:
                    probe.bind(('127.0.0.1', worker_ports[2]+1))
                self.ports.add(worker_ports[2]+1)
            coordinator = self.free_port()
            workers = ['127.0.0.1:'+str(port) for port in worker_ports]
            if cores == 2:
                workers.append('127.0.0.1:'+str(worker_ports[2]+1))
            agent_ports = [self.free_port() for _ in range(3)]
            try:
                for at, language in enumerate(('Java', 'CSharp', 'Python')):
                    args = argparse.Namespace(scenario=scenario, cores=cores, core_start=at*cores, language=language,
                                              worker_port=worker_ports[at], advertise='127.0.0.1', classes=str(self.artifacts),
                                              csharp=str(self.artifacts/'csharp/Worker.dll'), logs=str(self.artifacts/(scenario+'-nodes')),
                                              coordinator_port=coordinator, coordinator_workers=None)
                    node = node_agent.Node(args); nodes.append(node); node.launch()
                nodes[0].args.coordinator_workers = workers
                nodes[0].start(['java', '-Dopenaicubes.remotePostprocess=true', '-Dopenaicubes.denseChunkSize=1',
                                '-Dopenaicubes.benchmarkProfile='+scenario, '-cp', str(self.artifacts), 'CentralServer',
                                str(coordinator), *workers], coordinator, 'Coordinator', 0, nodes[0].mask)
                for node, port in zip(nodes, agent_ports):
                    def handler_factory(current):
                        class Handler(socketserver.BaseRequestHandler):
                            def handle(self):
                                fields = node_agent.receive(self.request).split('|')
                                node_agent.send(self.request, 'NODEINFO|'+fields[1]+'|'+current.snapshot())
                        return Handler
                    server = socketserver.ThreadingTCPServer(('127.0.0.1', port), handler_factory(node))
                    servers.append(server); threading.Thread(target=server.serve_forever, daemon=True).start()
                output, summary = self.benchmark('profile-'+scenario.lower(), scenario=scenario, requests=30, warmup=3,
                                                 clients=3, host='127.0.0.1', port=coordinator,
                                                 agents=','.join('127.0.0.1:'+str(port) for port in agent_ports),
                                                 **{'allow.localhost': 'true'})
                self.assertEqual(summary['success'], '30')
                self.assertEqual(summary['errors'], '0')
                self.assertEqual(summary['compute_slots'], str(3*cores))
                self.assertEqual(summary['compute_processes'], '3' if cores == 1 else '4')
                self.assertEqual(summary['observed_machines'], '1')
                self.assertEqual(summary['verification'], 'localhost_emulation')
                self.assertEqual(summary['physical_cluster_verified'], 'false')
                self.assertEqual(summary['client.affinity'], summary['before.node.0.affinity.mask'])
                self.assertEqual(summary['speedup.total'], '')
                self.assertIsNone(graphs.measured(output))
                self.assertIsNotNone(graphs.measured(output, allow_localhost=True))
                print('ETAPA6:', scenario, 'localhost only;', summary['compute_processes'], 'workers;', summary['compute_slots'], 'slots; 30/30', flush=True)
                original = graphs.properties(output.with_name(output.name+'.summary.properties'))
                config = self.artifacts/('profile-'+scenario.lower()+'.properties')
                rejected = subprocess.run(['java', '-cp', str(self.artifacts), 'benchmarks.Benchmark', '--config', str(config),
                                           'allow.localhost=false', 'requests=1'], cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn('three_distinct_machines_required', rejected.stderr)
                self.assertEqual(original['configuration_valid'], 'true')
                trace = (self.artifacts/(scenario+'-nodes')/('Coordinator-'+str(coordinator)+'.log')).read_text(encoding='utf-8')
                self.assertIn('|POST|', trace)
                for language in ('Java', 'Python', 'CSharp'):
                    self.assertIn('|'+language+'|', trace)
                if scenario == 'DISTRIBUTED':
                    process = nodes[2].processes[0][0]
                    resource_probe.set_affinity(process.pid, nodes[0].mask)
                    self.assertIn('valid=false', nodes[2].snapshot())
            finally:
                for server in servers:
                    server.shutdown(); server.server_close()
                for node in reversed(nodes):
                    node.close()


def load_tests(loader, tests, pattern):
    names = [name for name in loader.getTestCaseNames(BenchmarkTests) if name in BenchmarkTests.__dict__]
    return unittest.TestSuite(BenchmarkTests(name) for name in names)


if __name__ == '__main__':
    unittest.main(verbosity=2)
