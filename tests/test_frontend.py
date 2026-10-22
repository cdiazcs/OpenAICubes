"""ETAPA 4: Swing headless + TCP real, sin dependencias. python tests/test_frontend.py

Las capturas headless verifican el render de componentes, no una ventana abierta.
FrontendChecks flow ... --visual es una comprobacion separada con JFrame real.
"""
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import unittest

import test_protocol as protocol
from test_distributed import Forwarder
from test_identify import FaultPeer, Drip, STALL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_python import CNN
from cnn_python.images import PNG, encode_png, read_image


def png_chunk(kind, body):
    import zlib
    return struct.pack('!I', len(body))+kind+body+struct.pack('!I', zlib.crc32(kind+body) & 0xffffffff)


class FrontendTests(protocol.ProtocolTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.compile(['javac', '--release', '8', '-encoding', 'UTF-8', '-cp', str(cls.artifacts),
                     '-d', str(cls.artifacts), 'clients-java/ClientCli.java']
                    + [str(p) for p in sorted((ROOT/'frontend-java').glob('*.java'))]
                    + ['tests/FrontendChecks.java'], 'javac-frontend')
        cls.model = CNN.load()

    def java(self, *args, timeout=35):
        result = subprocess.run(['java', '-Djava.awt.headless=true', '-cp', str(self.artifacts),
                                 'FrontendChecks', *map(str, args)], cwd=ROOT, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=timeout)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result.stdout

    def test_unit_components_ids_messages_jpeg_and_invalid_formats(self):
        result = self.java('unit', self.artifacts/'ui-unit')
        self.assertIn('UNIT_OK|1600_unique_ids', result)

    def test_shared_preprocessing_every_pixel_png_pgm_alpha_and_resize(self):
        import zlib
        fixtures = []
        for width, height in ((16, 16), (31, 23), (7, 9)):
            path = self.artifacts/('gray-'+str(width)+'.png')
            path.write_bytes(encode_png(width, height, [(i*29+7) % 256 for i in range(width*height)]))
            fixtures.append(path)
        for color, channels in ((2, 3), (6, 4)):
            width, height = 21, 17
            rows = [b'\0'+bytes((x*17+y*11+c*53) % 256 for x in range(width) for c in range(channels))
                    for y in range(height)]
            body = PNG+png_chunk(b'IHDR', struct.pack('!IIBBBBB', width, height, 8, color, 0, 0, 0))
            body += png_chunk(b'IDAT', zlib.compress(b''.join(rows)))+png_chunk(b'IEND', b'')
            path = self.artifacts/('color-'+str(color)+'.png'); path.write_bytes(body); fixtures.append(path)
        path = self.artifacts/'gray.pgm'; path.write_bytes(b'P5\n7 9\n255\n'+bytes((i*5) % 256 for i in range(63))); fixtures.append(path)
        for path in fixtures:
            with self.subTest(image=path.name):
                output = self.java('image', path)
                actual = [float(value) for value in output.strip().split('|')[1].split(',')]
                expected = read_image(path)
                self.assertEqual(len(actual), 256)
                for a, b in zip(actual, expected): self.assertAlmostEqual(a, b, delta=1e-12)

    def test_image_client_both_modes_real_server_three_classes(self):
        for mode in ('SEQUENTIAL', 'DISTRIBUTED'):
            for name in ('circulo', 'cruz', 'triangulo'):
                image = ROOT/'model/examples'/(name+'-0.png')
                result = self.java('request', '127.0.0.1', self.central, mode, image, 15000)
                reply = next(line for line in result.splitlines() if line.startswith('RESULT|'))
                request_id = reply.split('|')[1]
                self.result(reply, request_id, 1, 3, self.model.predict(read_image(image)).probabilities)
                self.assertIn('CLASS|'+name, result)
                self.assertEqual([line for line in result.splitlines() if line.startswith('PHASE|')],
                                 ['PHASE|CONNECTING', 'PHASE|SENDING', 'PHASE|PROCESSING', 'PHASE|RECEIVED'])

    def test_swing_concurrent_out_of_order_real_inference_and_edt(self):
        def delay(fields):
            if fields[0] == 'IDENTIFY': time.sleep(.6 if fields[3] == 'SEQUENTIAL' else .12)
        with Forwarder(self.central, before=delay) as proxy:
            result = self.java('flow', '127.0.0.1', proxy.port, ROOT/'model/examples', self.artifacts/'ui-flow')
            self.assertIn('FLOW_OK|6|EDT_ticks=', result)
            self.assertIn('|visual=false', result)
            rows = [line.split('|') for line in result.splitlines() if line.startswith('FLOW|')]
            self.assertEqual(len(rows), 6)
            for row in rows:
                image = ROOT/'model/examples'/row[1]
                self.result('|'.join(row[3:]), row[4], 1, 3, self.model.predict(read_image(image)).probabilities)
            events = [event for event in proxy.events if event[2][0] == 'IDENTIFY']
            self.assertEqual(len([e for e in events if e[0] == 'start']), 6)
            self.assertEqual({e[2][3] for e in events}, {'SEQUENTIAL', 'DISTRIBUTED'})
            completed = [e[2][1] for e in events if e[0] == 'end']
            submitted = [e[2][1] for e in events if e[0] == 'start']
            self.assertNotEqual(completed, submitted)
            self.assertFalse(proxy.errors)

    def test_connection_refused_and_gui_can_retry(self):
        with socket.socket() as unavailable:
            unavailable.bind(('127.0.0.1', 0))
            result = self.java('error', '127.0.0.1', unavailable.getsockname()[1], ROOT/'model/examples/cruz-0.png', 1)
            self.assertIn('UI_ERROR|No se pudo conectar', result)

    def test_gui_timeout_and_cancellation_close_resources(self):
        with FaultPeer(lambda fields: STALL) as peer:
            result = self.java('error', '127.0.0.1', peer.port, ROOT/'model/examples/cruz-0.png', 1)
            self.assertIn('UI_ERROR|Se agotó el tiempo', result)
            result = self.java('error', '127.0.0.1', peer.port, ROOT/'model/examples/cruz-0.png', 15, 'cancel')
            self.assertIn('UI_ERROR|Consulta cancelada', result)

    def test_protocol_bad_replies_fragmentation_and_absolute_deadline(self):
        image = ROOT/'model/examples/cruz-0.png'
        def fragmented(fields):
            reply = 'RESULT|'+fields[1]+'|1|3|0.2,0.3,0.5'
            return Drip(protocol.frame(reply))
        with FaultPeer(fragmented) as peer:
            result = self.java('request', '127.0.0.1', peer.port, 'SEQUENTIAL', image, 5000)
            self.assertIn('|1|3|0.2,0.3,0.5', result)
        for response in (lambda f: 'RESULT|wrong|1|3|0.2,0.3,0.5',
                         lambda f: 'RESULT|'+f[1]+'|1|3|NaN,0,1',
                         lambda f: 'ERROR|'+f[1]+'|insufficient_workers',
                         lambda f: struct.pack('!I', 2_000_001),
                         lambda f: struct.pack('!I', 3)+b'\xff\xfe\xff'):
            with FaultPeer(response) as peer:
                result = self.java('error', '127.0.0.1', peer.port, image, 2)
                self.assertIn('UI_ERROR|', result)
        with FaultPeer(fragmented) as peer:
            result = self.java('error', '127.0.0.1', peer.port, image, 1)
            self.assertIn('UI_ERROR|Se agotó el tiempo', result)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(FrontendTests(name) for name in sorted(FrontendTests.__dict__)
                              if name.startswith('test_'))


def visual_smoke():
    """Abre la misma ventana de produccion, selecciona con JFileChooser y la cierra.

    Solo se ejecuta explicitamente: python tests/test_frontend.py --visual.
    No forma parte del discovery headless; requiere un escritorio grafico disponible.
    """
    FrontendTests.setUpClass()
    try:
        def delay(fields):
            if fields[0] == 'IDENTIFY': time.sleep(.4 if fields[3] == 'SEQUENTIAL' else .15)
        with Forwarder(FrontendTests.central, before=delay) as proxy:
            result = subprocess.run(['java', '-cp', str(FrontendTests.artifacts), 'FrontendChecks',
                                     'flow', '127.0.0.1', str(proxy.port), str(ROOT/'model/examples'),
                                     str(FrontendTests.artifacts/'ui-visual'), '--visual'],
                                    cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                                    errors='replace', timeout=50)
            print(result.stdout, end='', flush=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            if proxy.errors:
                raise RuntimeError(str(proxy.errors))
    finally:
        FrontendTests.doClassCleanups()


if __name__ == '__main__':
    if sys.argv[1:] == ['--visual']: visual_smoke()
    else: unittest.main(verbosity=2)
