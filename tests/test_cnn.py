"""Pruebas matematicas y de precision independientes, solo unittest/stdlib."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cnn_python.dataset import DEFAULT_DATASET, generate, load_split, read_manifest
from cnn_python.evaluate import evaluate
from cnn_python.images import PNG, Raster, decode_image, encode_png, normalize, read_image
from cnn_python.model import (CNN, DEFAULT_WEIGHTS, SHAPES, convolve_valid, flatten,
                              load_weights, loss_and_gradients, max_pooling, mlp,
                              pool_with_indices, relu, save_weights, softmax)
from cnn_python.train import initial_weights, train
from smoke_test import exchange


class MathTests(unittest.TestCase):
    def test_full_pipeline_dimensions_and_shared_convolution(self):
        weights = initial_weights(23)
        pixels = [i/255 for i in range(256)]
        state = CNN(weights).trace(pixels)
        self.assertEqual([len(channel) for channel in state['convolution']], [196, 196])
        self.assertEqual([len(channel) for channel in state['pooled']], [49, 49])
        self.assertEqual([len(state[key]) for key in ('flat', 'hidden', 'logits', 'probabilities')],
                         [98, 8, 3, 3])
        for channel, key in enumerate(('FILTER1', 'FILTER2')):
            expected, rows, cols = convolve_valid(pixels, 16, 16, weights[key], 3, 3)
            self.assertEqual((rows, cols), (14, 14))
            self.assertEqual(state['convolution'][channel], expected)
        self.assertEqual(state['flat'], state['pooled'][0]+state['pooled'][1])

    def test_convolution_known_values(self):
        self.assertEqual(convolve_valid(list(range(1, 10)), 3, 3, [1, 0, 0, -1], 2, 2),
                         ([-4, -4, -4, -4], 2, 2))
        self.assertEqual(convolve_valid(list(range(1, 10)), 3, 3, [1, 2, 3, 4], 2, 2),
                         ([37, 47, 67, 77], 2, 2))
        self.assertEqual(convolve_valid(list(range(1, 7)), 2, 3, [2, -1], 1, 2),
                         ([0, 1, 3, 4], 2, 2))

    def test_relu(self):
        self.assertEqual(relu([-2, -.5, 0, .3, 2]), [0, 0, 0, .3, 2])

    def test_max_pooling_and_tie_order(self):
        self.assertEqual(max_pooling(list(range(1, 17)), 4, 4), ([6, 8, 14, 16], 2, 2))
        self.assertEqual(pool_with_indices([1]*4, 2, 2), ([1], 1, 1, [0]))
        self.assertEqual(max_pooling(list(range(1, 10)), 3, 3), ([5], 1, 1))

    def test_flatten_channel_and_row_order(self):
        self.assertEqual(flatten([[1, 2, 3, 4], [5, 6, 7, 8]], 2, 2), list(range(1, 9)))

    def test_mlp_reference(self):
        values = mlp([1, 2], [1, 0, 0, 1], [0, 0], [1, 0, 0, 1], [0, 0])
        self.assertAlmostEqual(values[0], 1/(1+math.e), delta=1e-12)
        self.assertAlmostEqual(values[1], math.e/(1+math.e), delta=1e-12)
        values = mlp([-1, 2], [1, 0, 0, 1], [0, 0], [1, 0, 0, 1], [0, 0])
        self.assertAlmostEqual(values[0], 1/(1+math.exp(2)), delta=1e-12)

    def test_softmax_stability(self):
        self.assertEqual(softmax([1000, 1000, 1000]), [1/3]*3)
        self.assertEqual(softmax([-10000]), [1])
        values = softmax([1000, 1001, -1000])
        self.assertAlmostEqual(sum(values), 1, delta=1e-15)
        self.assertAlmostEqual(values[0], 1/(1+math.e), delta=1e-12)
        self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in values))

    def test_invalid_inputs(self):
        model = CNN(initial_weights(42))
        for pixels in ([0]*255, [0]*257, [-.1]*256, [1.1]*256,
                       [math.nan]*256, [math.inf]*256):
            with self.subTest(pixels=pixels[:1]), self.assertRaises(ValueError):
                model.predict(pixels)
        operations = [lambda: convolve_valid([1], 0, 1, [1], 1, 1),
                      lambda: convolve_valid([1], 1, 1, [1]*4, 2, 2),
                      lambda: convolve_valid([1]*3, 2, 2, [1], 1, 1),
                      lambda: convolve_valid([1e308], 1, 1, [2], 1, 1),
                      lambda: relu([math.nan]), lambda: softmax([]),
                      lambda: softmax([math.inf]), lambda: max_pooling([1], 1, 1),
                      lambda: flatten([[1], [2]], 2, 2),
                      lambda: mlp([1], [1, 2], [0], [1], [0])]
        for operation in operations:
            with self.assertRaises(ValueError):
                operation()

    def test_gradients_all_837_parameters(self):
        rng = random.Random(701)
        image = tuple(rng.uniform(.05, .95) for _ in range(256))
        weights = initial_weights(23)
        loss, analytic = loss_and_gradients(weights, image, 2)
        self.assertTrue(math.isfinite(loss) and loss > 0)
        checked, epsilon = 0, 1e-5
        for key in SHAPES:
            for index in range(SHAPES[key]):
                original = weights[key][index]
                weights[key][index] = original+epsilon
                positive = loss_and_gradients(weights, image, 2)[0]
                weights[key][index] = original-epsilon
                negative = loss_and_gradients(weights, image, 2)[0]
                weights[key][index] = original
                numeric = (positive-negative)/(2*epsilon)
                self.assertAlmostEqual(analytic[key][index], numeric, delta=2e-6,
                                       msg=key+'['+str(index)+']')
                checked += 1
        self.assertEqual(checked, 837)

    def test_weight_round_trip_and_immutable_model(self):
        weights = initial_weights(42)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'weights.txt'
            save_weights(weights, path)
            loaded = load_weights(path)
            self.assertEqual(loaded, {key: tuple(value) for key, value in weights.items()})
            first = path.read_bytes()
            save_weights(loaded, path)
            self.assertEqual(path.read_bytes(), first)
            path.write_text('FILTER1\n1,2,3\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_weights(path)
        model = CNN(weights)
        with self.assertRaises(TypeError):
            model.weights['B2'] = (0, 0, 0)
        weights['B2'][0] = 999
        self.assertEqual(model.weights['B2'][0], 0)
        for invalid in ({}, {**weights, 'UNKNOWN': [1]}, {**weights, 'FILTER1': [1]*8},
                        {**weights, 'B2': [math.nan, 0, 0]}):
            with self.assertRaises(ValueError):
                CNN(invalid)

    def test_tiny_training_reproducible_and_all_layers_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'tiny.zip'
            generate(path, 2, 1, 1)
            training, validation = load_split(path, 'train'), load_split(path, 'validation')
            first, report = train(training, validation, epochs=2, batch_size=3, seed=23, verbose=False)
            second, _ = train(training, validation, epochs=2, batch_size=3, seed=23, verbose=False)
            self.assertEqual(first, second)
            self.assertTrue(all(value > 0 for value in report['parameter_change_l2'].values()))


class ImageTests(unittest.TestCase):
    def test_normalization_and_resize(self):
        raster = Raster(2, 2, (0, 255, 127.5, 64))
        self.assertEqual(normalize(raster, 2), (0, 1, .5, 64/255))
        resized = normalize(raster)
        self.assertEqual(len(resized), 256)
        self.assertEqual(resized[0], 0)
        self.assertEqual(resized[15], 1)
        self.assertEqual(resized[-1], 64/255)

    def test_png_round_trip_and_content_based_format(self):
        pixels = list(range(256))
        png = encode_png(16, 16, pixels)
        self.assertEqual(normalize(decode_image(png)), tuple(x/255 for x in pixels))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'sin_clase.dat'
            path.write_bytes(png)
            self.assertEqual(read_image(path), tuple(x/255 for x in pixels))

    def test_png_filters_and_rgb(self):
        def chunk(kind, body):
            return struct.pack('!I', len(body))+kind+body+struct.pack('!I', zlib.crc32(kind+body))

        # Dos filas gris conocidas; se codifican con cada filtro de PNG.
        fixtures = {0: bytes([0, 10, 20, 30, 0, 40, 50, 60]),
                    1: bytes([1, 10, 10, 10, 1, 40, 10, 10]),
                    2: bytes([2, 10, 20, 30, 2, 30, 30, 30]),
                    3: bytes([3, 10, 15, 20, 3, 35, 20, 20]),
                    4: bytes([4, 10, 10, 10, 4, 30, 10, 10])}
        for mode, raw in fixtures.items():
            header = struct.pack('!IIBBBBB', 3, 2, 8, 0, 0, 0, 0)
            png = PNG+chunk(b'IHDR', header)+chunk(b'IDAT', zlib.compress(raw))+chunk(b'IEND', b'')
            with self.subTest(filter=mode):
                self.assertEqual(decode_image(png).pixels, (10, 20, 30, 40, 50, 60))
        header = struct.pack('!IIBBBBB', 2, 1, 8, 2, 0, 0, 0)
        png = PNG+chunk(b'IHDR', header)+chunk(b'IDAT', zlib.compress(bytes([0, 255, 0, 0, 0, 255, 0])))+chunk(b'IEND', b'')
        self.assertEqual(decode_image(png).pixels, (76.245, 149.685))

    def test_pgm_text_binary_and_comments(self):
        self.assertEqual(decode_image(b'P2\n# test\n2 1\n100\n0 100\n').pixels, (0, 255))
        self.assertEqual(decode_image(b'P5\r\n2 1\r\n255\r\n'+bytes([10, 32])).pixels, (10, 32))

    def test_invalid_images(self):
        png = encode_png(1, 1, [25])
        for data in (b'', b'not an image', png[:-1], png[:30]+b'x'+png[31:],
                     b'P5\n2 2\n255\nabc', b'P2\n1 1\n255\n256', b'P2\n1 1\n255\n0 1'):
            with self.subTest(data=data[:8]), self.assertRaises(ValueError):
                decode_image(data)
        with self.assertRaises(ValueError):
            normalize(Raster(1, 1, (math.nan,)))


class DatasetTests(unittest.TestCase):
    def test_independent_balanced_splits(self):
        manifest = read_manifest()
        self.assertEqual(len(set(manifest['seeds'].values())), 3)
        digests = {split: {entry['pixels_sha256'] for entry in manifest['entries']
                           if entry['split'] == split} for split in manifest['seeds']}
        self.assertFalse(digests['train'] & digests['test'])
        self.assertFalse(digests['validation'] & digests['test'])
        self.assertFalse(digests['train'] & digests['validation'])

    def test_dataset_generation_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary)/'a.zip', Path(temporary)/'b.zip'
            generate(first, 2, 1, 1)
            generate(second, 2, 1, 1)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(len(load_split(first, 'train')), 6)


class AcceptanceTests(unittest.TestCase):
    def test_independent_accuracy_and_persisted_confusion_matrix(self):
        report, predictions = evaluate(CNN.load(), load_split(split='test'))
        self.assertGreaterEqual(report['accuracy'], .85, report)
        self.assertEqual(report['samples'], 300)
        self.assertEqual(len(predictions), 300)
        persisted = json.loads((ROOT/'model/evaluation/metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(report['accuracy'], persisted['accuracy'])
        self.assertEqual(report['confusion_matrix_rows_true_columns_predicted'],
                         persisted['confusion_matrix_rows_true_columns_predicted'])
        self.assertEqual(hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest(), persisted['weights_sha256'])
        self.assertEqual(hashlib.sha256(DEFAULT_DATASET.read_bytes()).hexdigest(), persisted['dataset_sha256'])

    def test_pixels_determine_prediction_and_filename_does_not(self):
        model = CNN.load()
        for label, name in enumerate(('circulo', 'cruz', 'triangulo')):
            path = ROOT/'model/examples'/(name+'-0.png')
            self.assertEqual(model.predict(read_image(path)).index, label)
            with tempfile.TemporaryDirectory() as temporary:
                renamed = Path(temporary)/'otra_clase.bin'
                renamed.write_bytes(path.read_bytes())
                self.assertEqual(model.predict(read_image(path)), model.predict(read_image(renamed)))

    def test_inference_weights_from_another_directory(self):
        code = ('import sys; sys.path.insert(0, sys.argv[1]); '
                'from cnn_python import CNN; from cnn_python.images import read_image; '
                'print(CNN.load().predict(read_image(sys.argv[2])).label)')
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, '-c', code, str(ROOT),
                                     str(ROOT/'model/examples/circulo-0.png')], cwd=temporary,
                                    capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), b'circulo')

    def test_inference_does_not_open_sockets(self):
        with mock.patch('socket.socket', side_effect=AssertionError('unexpected socket')):
            self.assertEqual(CNN.load().predict(read_image(ROOT/'model/examples/cruz-0.png')).label, 'cruz')


class WorkerIntegrationTests(unittest.TestCase):
    def test_atomic_cold_load_matches_local_inference(self):
        from concurrent.futures import ThreadPoolExecutor
        spec = importlib.util.spec_from_file_location('worker_stage1', ROOT/'workers/python/worker.py')
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        pixels = read_image(ROOT/'model/examples/triangulo-0.png')
        image_csv = ','.join(str(value) for value in pixels)
        expected = CNN.load().predict(pixels).probabilities
        original = worker.CNN.load

        def slow_load(path):
            time.sleep(.02)
            return original(path)

        with mock.patch.object(worker.CNN, 'load', side_effect=slow_load) as loader:
            with ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(worker.run_full_cnn, [image_csv]*36))
            self.assertEqual(loader.call_count, 1)
        self.assertTrue(all(tuple(result) == expected for result in results))

    def test_worker_from_other_directory_and_identify_validation(self):
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        with tempfile.TemporaryDirectory() as temporary:
            with (Path(temporary)/'worker.log').open('wb') as log:
                process = subprocess.Popen([sys.executable, str(ROOT/'workers/python/worker.py'), str(port)],
                                           cwd=temporary, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic()+10
                    while True:
                        if process.poll() is not None:
                            self.fail('worker terminated')
                        try:
                            self.assertEqual(exchange('127.0.0.1', port, 'PING|ready'), 'PONG|ready|Python')
                            break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise
                            time.sleep(.05)
                    pixels = read_image(ROOT/'model/examples/circulo-0.png')
                    reply = exchange('127.0.0.1', port, 'IDENTIFY|image|'+','.join(map(str, pixels)))
                    fields = reply.split('|')
                    self.assertEqual(fields[:4], ['RESULT', 'image', '1', '3'])
                    expected = CNN.load().predict(pixels).probabilities
                    self.assertTrue(all(abs(float(a)-b) < 1e-8 for a, b in zip(fields[4].split(','), expected)))
                    reply = exchange('127.0.0.1', port, 'IDENTIFY|invalid|'+','.join(['2']*256))
                    self.assertTrue(reply.startswith('ERROR|invalid|'), reply)
                finally:
                    process.terminate()
                    process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
