
import argparse
import hashlib
import os
from pathlib import Path
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time

from resource_probe import topology, affinity, set_affinity, select_cores, properties

ROOT = Path(__file__).resolve().parents[1]


def exact(connection, count):
    result = bytearray()
    while len(result) < count:
        part = connection.recv(count-len(result))
        if not part:
            raise EOFError('truncated_frame')
        result.extend(part)
    return bytes(result)


def receive(connection):
    size = struct.unpack('!I', exact(connection, 4))[0]
    if not 0 < size <= 2_000_000:
        raise ValueError('invalid_frame')
    return exact(connection, size).decode('utf-8')


def send(connection, message):
    body = message.encode('utf-8')
    connection.sendall(struct.pack('!I', len(body))+body)


def rpc(host, port, message):
    with socket.create_connection((host, port), timeout=3) as connection:
        connection.settimeout(3)
        send(connection, message)
        return receive(connection)


class Node:
    def __init__(self, args):
        self.args, self.processes, self.logs, self.violations = args, [], [], []
        self.stop = threading.Event()
        self.info = topology()
        self.cores = select_cores(self.info, args.cores, args.core_start, affinity(os.getpid()))
        self.mask = sum(mask for _, mask in self.cores)
        self.weights_hash = hashlib.sha256((ROOT/'model/weights/weights.txt').read_bytes()).hexdigest()

    def start(self, command, port, role, slots, mask):
        path = Path(self.args.logs)
        path.mkdir(parents=True, exist_ok=True)
        log = (path/(role+'-'+str(port)+'.log')).open('w', encoding='utf-8')
        self.logs.append(log)
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        self.processes.append((process, port, role, slots, mask))
        set_affinity(process.pid, mask)
        end = time.monotonic()+25
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError('process_exited: '+str(log.name))
            try:
                if role == 'Coordinator':
                    status = rpc('127.0.0.1', port, 'CONFIG|agent-ready').split('|')
                    ready = len(status) == 8 and status[:4] == ['CONFIG', 'agent-ready', self.args.scenario, 'true']
                else:
                    status = rpc('127.0.0.1', port, 'CAPS|agent-ready').split('|')
                    ready = len(status) == 5 and status[:3] == ['CAPS', 'agent-ready', role] and int(status[4]) == slots
                if ready:
                    return
            except (OSError, EOFError):
                time.sleep(.05)
        raise RuntimeError('process_not_ready: '+str(log.name))

    def launch(self):
        args = self.args
        threads = 1 if args.scenario == 'DISTRIBUTED' else args.cores
        if args.language == 'Python':
            for index, (_, mask) in enumerate(self.cores):
                self.start([sys.executable, 'workers/python/worker.py', str(args.worker_port+index), '1'],
                           args.worker_port+index, 'Python', 1, mask)
        elif args.language == 'Java':
            self.start(['java', '-cp', args.classes, 'WorkerJava', str(args.worker_port), str(threads)],
                       args.worker_port, 'Java', threads, self.mask)
        else:
            self.start(['dotnet', args.csharp, str(args.worker_port), str(threads)],
                       args.worker_port, 'CSharp', threads, self.mask)
        if args.coordinator_workers:
            self.start(['java', '-Dopenaicubes.remotePostprocess=true',
                        '-Dopenaicubes.benchmarkProfile='+args.scenario,
                        '-Dopenaicubes.taskThreads='+str(max(8, args.cores*3)),
                        '-Dopenaicubes.denseChunkSize=1', '-cp', args.classes, 'CentralServer',
                        str(args.coordinator_port), *args.coordinator_workers],
                       args.coordinator_port, 'Coordinator', 0, self.mask)
        self.monitor = threading.Thread(target=self.watch, daemon=True)
        self.monitor.start()

    def watch(self):
        while not self.stop.wait(.1):
            for process, port, role, _, mask in self.processes:
                try:
                    if process.poll() is not None or affinity(process.pid) != mask:
                        issue = role+':process_or_affinity_changed:'+str(port)
                        if issue not in self.violations:
                            self.violations.append(issue)
                except OSError:
                    self.violations.append(role+':probe_failed')

    def snapshot(self):
        data = {'version': 6, 'scenario': self.args.scenario, 'machine.id': self.info['machine_id'],
                'machine.name': self.info['hostname'], 'machine.os': self.info['os'],
                'cpu.name': self.info['cpu'], 'cpu.physical.total': self.info['physical_cores'],
                'cpu.logical.total': self.info['logical_cpus'], 'budget.cores': self.args.cores,
                'budget.compute_threads': self.args.cores, 'weights.sha256': self.weights_hash,
                'affinity.mask': format(self.mask, 'x'), 'core.ids': ','.join(str(core) for core, _ in self.cores),
                'agent.pid': os.getpid(), 'agent.role': 'control_only', 'agent.affinity.actual': format(affinity(os.getpid()), 'x'),
                'physical_cluster_verified': 'false'}
        workers = [p for p in self.processes if p[2] != 'Coordinator']
        data['workers.count'] = len(workers)
        for index, (process, port, role, slots, mask) in enumerate(workers):
            prefix = 'worker.'+str(index)+'.'
            actual = affinity(process.pid) if process.poll() is None else 0
            caps = rpc('127.0.0.1', port, 'CAPS|evidence').split('|') if actual else []
            if len(caps) != 5 or caps[2] != role or int(caps[4]) != slots or 'POST' not in caps[3].split(','):
                self.violations.append(role+':invalid_caps')
            data.update({prefix+'endpoint': self.args.advertise+':'+str(port), prefix+'pid': process.pid,
                         prefix+'language': role, prefix+'slots': slots, prefix+'affinity.expected': format(mask, 'x'),
                         prefix+'affinity.actual': format(actual, 'x'), prefix+'alive': str(process.poll() is None).lower()})
        for process, port, role, _, mask in self.processes:
            if role == 'Coordinator':
                actual = affinity(process.pid) if process.poll() is None else 0
                data.update({'coordinator.endpoint': self.args.advertise+':'+str(port),
                             'coordinator.pid': process.pid, 'coordinator.affinity': format(actual, 'x'),
                             'coordinator.registered': ','.join(self.args.coordinator_workers)})
                if actual != mask:
                    self.violations.append('coordinator:affinity_changed')
        data['violations'] = ','.join(sorted(set(self.violations)))
        data['valid'] = str(not self.violations and all(affinity(p.pid) == m for p, _, _, _, m in self.processes if p.poll() is None)
                            and all(p.poll() is None for p, *_ in self.processes)).lower()
        return properties(data)

    def close(self):
        self.stop.set()
        for process, *_ in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
        for process, *_ in reversed(self.processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        for log in self.logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['DISTRIBUTED', 'PARALLEL_DISTRIBUTED'], required=True)
    parser.add_argument('--language', choices=['Java', 'Python', 'CSharp'], required=True)
    parser.add_argument('--cores', type=int, required=True)
    parser.add_argument('--core-start', type=int, default=0)
    parser.add_argument('--worker-port', type=int, required=True)
    parser.add_argument('--agent-port', type=int, default=9200)
    parser.add_argument('--advertise', required=True, help='IP real de esta laptop, accesible desde el benchmark')
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--classes', default='build')
    parser.add_argument('--csharp', default='build/csharp/Worker.dll')
    parser.add_argument('--logs', default='build/stage6/node-logs')
    parser.add_argument('--coordinator-port', type=int, default=9000)
    parser.add_argument('--coordinator-workers', nargs='+')
    args = parser.parse_args()
    if args.cores < 1 or args.cores > 32 or (args.scenario == 'DISTRIBUTED' and args.cores != 1) or (args.scenario == 'PARALLEL_DISTRIBUTED' and args.cores < 2):
        parser.error('DISTRIBUTED requiere 1 core; PARALLEL_DISTRIBUTED requiere 2..32')
    node = Node(args)
    set_affinity(os.getpid(), node.mask)
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(5)
            try:
                fields = receive(self.request).split('|')
                if len(fields) == 2 and fields[0] == 'NODEINFO' and 0 < len(fields[1]) <= 128:
                    send(self.request, 'NODEINFO|'+fields[1]+'|'+node.snapshot())
            except (OSError, EOFError, ValueError):
                return
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    try:
        node.launch()
        with Server((args.bind, args.agent_port), Handler) as server:
            print('NODE_AGENT_READY|'+args.advertise+':'+str(args.agent_port), flush=True)
            server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()


if __name__ == '__main__':
    main()
