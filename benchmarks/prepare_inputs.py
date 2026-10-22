import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def prepare(dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    expected = set()
    with zipfile.ZipFile(dataset) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        entries = [entry for entry in manifest['entries'] if entry['split'] == 'test']
        for index, entry in enumerate(entries):
            name = '%06d.pgm' % index
            data = archive.read(entry['path'])
            if hashlib.sha256(data.split(b'\n', 3)[3]).hexdigest() != entry['pixels_sha256']:
                raise ValueError('dataset_checksum')
            (output/name).write_bytes(data)
            expected.add(name)
    actual = {path.name for path in output.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError('output_contains_extra_images')
    print('IDENTIFY inputs:', len(expected), 'output:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('model/data/geometry-v1.zip'))
    parser.add_argument('--output', type=Path, default=Path('build/stage6/images'))
    args = parser.parse_args()
    prepare(args.dataset, args.output)
