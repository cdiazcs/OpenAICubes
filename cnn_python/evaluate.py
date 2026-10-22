
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .dataset import DEFAULT_DATASET, ROOT, load_split, read_manifest
from .model import CLASSES, CNN, DEFAULT_WEIGHTS


def evaluate(model, examples):
    confusion = [[0]*3 for _ in range(3)]
    predictions = []
    for index, (pixels, expected) in enumerate(examples):
        prediction = model.predict(pixels)
        confusion[expected][prediction.index] += 1
        predictions.append({'sample': index, 'expected': CLASSES[expected],
                            'predicted': prediction.label,
                            **dict(zip(CLASSES, prediction.probabilities))})
    total = len(examples)
    if not total:
        raise ValueError('empty_evaluation')
    return {'classes': CLASSES, 'samples': total,
            'accuracy': sum(confusion[i][i] for i in range(3))/total,
            'confusion_matrix_rows_true_columns_predicted': confusion,
            'per_class_recall': {CLASSES[i]: confusion[i][i]/sum(confusion[i])
                                 if sum(confusion[i]) else None for i in range(3)}}, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--output', type=Path, default=ROOT / 'model' / 'evaluation')
    args = parser.parse_args()
    report, predictions = evaluate(CNN.load(args.weights), load_split(args.dataset, 'test'))
    report.update({'split': 'test', 'seed': read_manifest(args.dataset)['seeds']['test'],
                   'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                   'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                   'minimum_accuracy': .85, 'meets_target': report['accuracy'] >= .85})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output / 'predictions.csv').open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
