
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import zipfile

from .images import decode_image, encode_png, normalize
from .model import CLASSES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / 'model' / 'data' / 'geometry-v1.zip'
SPLIT_SEEDS = {'train': 1729, 'validation': 2718, 'test': 31415}
GENERATOR = 'geometry-v1'


def _segment_distance(x, y, ax, ay, bx, by):
    dx, dy = bx-ax, by-ay
    t = max(0, min(1, ((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy)))
    return math.hypot(x-ax-t*dx, y-ay-t*dy)


def sample(label, rng):
    if type(label) is not int or not 0 <= label < 3:
        raise ValueError('class_index')
    cx, cy = 7.5+rng.uniform(-1.25, 1.25), 7.5+rng.uniform(-1.25, 1.25)
    radius, thickness = rng.uniform(3.7, 5.3), rng.uniform(.9, 1.7)
    angle = rng.uniform(-.16, .16)
    foreground, background = rng.uniform(.8, 1), rng.uniform(0, .06)
    noise = rng.uniform(.015, .07)
    cosine, sine = math.cos(angle), math.sin(angle)
    vertices = [(radius*math.cos(-math.pi/2+i*2*math.pi/3),
                 radius*math.sin(-math.pi/2+i*2*math.pi/3)) for i in range(3)]

    def ink(x, y):
        dx, dy = x-cx, y-cy
        x, y = dx*cosine+dy*sine, -dx*sine+dy*cosine
        if label == 0:
            distance = abs(math.hypot(x, y)-radius)
        elif label == 1:
            distance = min(_segment_distance(x, y, -radius, 0, radius, 0),
                           _segment_distance(x, y, 0, -radius, 0, radius))
        else:
            distance = min(_segment_distance(x, y, *vertices[i], *vertices[(i+1) % 3])
                           for i in range(3))
        return distance <= thickness/2

    pixels = []
    for r in range(16):
        for c in range(16):
            coverage = sum(ink(c+dx, r+dy) for dx in (-.25, .25) for dy in (-.25, .25))/4
            value = background+(foreground-background)*coverage+rng.gauss(0, noise)
            if rng.random() < .002:
                value = rng.random()
            pixels.append(round(255*max(0, min(1, value))))
    return pixels


def _write(archive, name, body):
  
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, body, compresslevel=9)


def generate(path=DEFAULT_DATASET, train_per_class=240, validation_per_class=60,
             test_per_class=100, seeds=None, examples=None):
    seeds = dict(SPLIT_SEEDS if seeds is None else seeds)
    counts = dict(zip(SPLIT_SEEDS, (train_per_class, validation_per_class, test_per_class)))
    if (set(seeds) != set(SPLIT_SEEDS) or len(set(seeds.values())) != 3
            or any(type(n) is not int or n < 1 for n in counts.values())):
        raise ValueError('dataset_splits')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'version': GENERATOR, 'classes': CLASSES, 'width': 16, 'height': 16,
                'seeds': seeds, 'per_class': counts, 'entries': []}
    seen = set()
    with zipfile.ZipFile(path, 'w') as archive:
        for split, count in counts.items():
            rng = random.Random(seeds[split])
            for label in range(3):
                for index in range(count):
                    pixels = sample(label, rng)
                    body = b'P5\n16 16\n255\n' + bytes(pixels)
                    digest = hashlib.sha256(bytes(pixels)).hexdigest()
                    if digest in seen:
                        raise ValueError('duplicate_pixels_across_dataset')
                    seen.add(digest)
                    name = split + '/' + str(label*count+index).zfill(6) + '.pgm'
                    _write(archive, name, body)
                    manifest['entries'].append({'path': name, 'split': split,
                                                'label': label, 'pixels_sha256': digest})
                    if examples is not None and split == 'test' and index < 3:
                        target = Path(examples) / (CLASSES[label]+'-'+str(index)+'.png')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(encode_png(16, 16, pixels))
        _write(archive, 'manifest.json', json.dumps(manifest, sort_keys=True, indent=2).encode())
    return manifest


def read_manifest(path=DEFAULT_DATASET):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
    if (manifest.get('version') != GENERATOR or manifest.get('classes') != list(CLASSES)
            or (manifest.get('width'), manifest.get('height')) != (16, 16)
            or set(manifest.get('seeds', {})) != set(SPLIT_SEEDS)
            or len(set(manifest['seeds'].values())) != 3):
        raise ValueError('dataset_manifest')
    seen, counts = set(), {split: [0]*3 for split in SPLIT_SEEDS}
    for entry in manifest['entries']:
        split, label, digest = entry['split'], entry['label'], entry['pixels_sha256']
        if split not in counts or type(label) is not int or not 0 <= label < 3 or digest in seen:
            raise ValueError('dataset_labels_or_duplicates')
        seen.add(digest)
        counts[split][label] += 1
    if any(count != manifest['per_class'][split] for split in counts for count in counts[split]):
        raise ValueError('dataset_counts')
    return manifest


def load_split(path=DEFAULT_DATASET, split='train'):
    if split not in SPLIT_SEEDS:
        raise ValueError('dataset_split')
    manifest = read_manifest(path)
    result = []
    with zipfile.ZipFile(path) as archive:
        for entry in manifest['entries']:
            if entry['split'] == split:
                raster = decode_image(archive.read(entry['path']))
                if (raster.width, raster.height) != (16, 16):
                    raise ValueError('dataset_image_shape')
                if hashlib.sha256(bytes(round(x) for x in raster.pixels)).hexdigest() != entry['pixels_sha256']:
                    raise ValueError('dataset_checksum')
                result.append((normalize(raster), entry['label']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--examples', type=Path, default=ROOT / 'model' / 'examples')
    parser.add_argument('--train-per-class', type=int, default=240)
    parser.add_argument('--validation-per-class', type=int, default=60)
    parser.add_argument('--test-per-class', type=int, default=100)
    args = parser.parse_args()
    generate(args.output, args.train_per_class, args.validation_per_class, args.test_per_class,
             examples=args.examples)
    print('Dataset:', args.output)
    print('Semillas:', SPLIT_SEEDS)


if __name__ == '__main__':
    main()
