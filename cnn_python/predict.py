
import argparse
import json
from pathlib import Path
import time

from .images import read_image
from .model import CLASSES, CNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    result = CNN.load(args.weights).predict(read_image(args.image))
    print(json.dumps({'class_index': result.index, 'class': result.label,
                      'probabilities': dict(zip(CLASSES, result.probabilities)),
                      'elapsed_ms': 1000*(time.perf_counter()-started)}, indent=2))


if __name__ == '__main__':
    main()
