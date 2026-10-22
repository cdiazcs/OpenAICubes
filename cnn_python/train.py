
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

from .dataset import DEFAULT_DATASET, ROOT, load_split, read_manifest
from .model import CNN, SHAPES, finite, loss_and_gradients, save_weights


def initial_weights(seed):
    rng = random.Random(seed)
    weights = {}
    for key, count in SHAPES.items():
        fan_in = 9 if key.startswith('FILTER') else 98 if key == 'W1' else 8
        if key.startswith('B'):
            weights[key] = [0.05 if key == 'B1' else 0.0]*count
        else:
            weights[key] = [rng.gauss(0, math.sqrt(2/fan_in)) for _ in range(count)]
    return weights


def metrics(model, examples):
    correct, total_loss = 0, 0.0
    for image, label in examples:
        state = model.trace(image)
        logits = state['logits']
        maximum = max(logits)
        total_loss += maximum + math.log(sum(math.exp(x-maximum) for x in logits)) - logits[label]
        correct += max(range(3), key=state['probabilities'].__getitem__) == label
    return {'accuracy': correct/len(examples), 'cross_entropy': total_loss/len(examples)}


def train(training, validation, epochs=60, batch_size=24, learning_rate=.003,
          seed=20261022, patience=12, verbose=True):
    if (not training or not validation or epochs < 1 or batch_size < 1 or patience < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('training_parameters')
    weights = initial_weights(seed)
    starting = {key: values[:] for key, values in weights.items()}
    moment1 = {key: [0.0]*count for key, count in SHAPES.items()}
    moment2 = {key: [0.0]*count for key, count in SHAPES.items()}
    rng = random.Random(seed+1)
    history, best_loss, best_epoch, step = [], math.inf, 0, 0
    best = None
    initial_metrics = metrics(CNN(weights), validation)
    for epoch in range(1, epochs+1):
        order = list(range(len(training)))
        rng.shuffle(order)
        loss_total = 0.0
        for offset in range(0, len(order), batch_size):
            batch = order[offset:offset+batch_size]
            accumulated = {key: [0.0]*count for key, count in SHAPES.items()}
            for index in batch:
                image, label = training[index]
                loss, gradients = loss_and_gradients(weights, image, label)
                loss_total += loss
                for key in SHAPES:
                    source, target = gradients[key], accumulated[key]
                    for i in range(len(source)):
                        target[i] += source[i]/len(batch)
            norm = math.sqrt(sum(value*value for values in accumulated.values() for value in values))
            scale = min(1.0, 5.0/max(norm, 1e-30))
            step += 1
            for key in SHAPES:
                values, first, second = weights[key], moment1[key], moment2[key]
                for i, value in enumerate(accumulated[key]):
                    gradient = finite(value*scale)
                    first[i] = .9*first[i] + .1*gradient
                    second[i] = .999*second[i] + .001*gradient*gradient
                    estimate = first[i]/(1-.9**step)
                    variance = second[i]/(1-.999**step)
                    values[i] = finite(values[i] - learning_rate*estimate/(math.sqrt(variance)+1e-8))
        validation_metrics = metrics(CNN(weights), validation)
        row = {'epoch': epoch, 'train_cross_entropy': loss_total/len(training),
               'validation_cross_entropy': validation_metrics['cross_entropy'],
               'validation_accuracy': validation_metrics['accuracy']}
        history.append(row)
        if row['validation_cross_entropy'] < best_loss:
            best_loss, best_epoch = row['validation_cross_entropy'], epoch
            best = {key: values[:] for key, values in weights.items()}
        if verbose:
            print('epoch={:02d} train_loss={:.5f} val_loss={:.5f} val_acc={:.2%}'.format(
                epoch, row['train_cross_entropy'], row['validation_cross_entropy'], row['validation_accuracy']), flush=True)
        if epoch-best_epoch >= patience:
            break
    return best, {'seed': seed, 'epochs_requested': epochs, 'epochs_run': len(history),
                  'batch_size': batch_size, 'learning_rate': learning_rate, 'patience': patience,
                  'optimizer': 'Adam beta1=0.9 beta2=0.999 epsilon=1e-8; global gradient clip=5',
                  'selection': 'minimum validation cross entropy; no test samples loaded',
                  'best_epoch': best_epoch, 'initial_validation': initial_metrics,
                  'trained_parameters': list(SHAPES), 'history': history,
                  'parameter_change_l2': {key: math.sqrt(sum((a-b)**2 for a, b in zip(starting[key], best[key])))
                                          for key in SHAPES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--weights', type=Path, default=ROOT / 'model' / 'weights' / 'weights.txt')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'model' / 'training')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--learning-rate', type=float, default=.003)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--patience', type=int, default=12)
    args = parser.parse_args()
    started = time.perf_counter()
    training, validation = load_split(args.dataset, 'train'), load_split(args.dataset, 'validation')
    weights, report = train(training, validation, args.epochs, args.batch_size,
                            args.learning_rate, args.seed, args.patience)
    save_weights(weights, args.weights)
    report.update({'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                   'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                   'dataset_seeds': read_manifest(args.dataset)['seeds'],
                   'training_samples': len(training), 'validation_samples': len(validation),
                   'final_training': metrics(CNN(weights), training),
                   'final_validation': metrics(CNN(weights), validation),
                   'seconds': time.perf_counter()-started, 'python': platform.python_version(),
                   'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                      for name in ('model.py', 'dataset.py', 'train.py')}})
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / 'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.report_dir / 'history.csv').open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(report['history'][0]))
        writer.writeheader()
        writer.writerows(report['history'])
    print('Pesos:', args.weights, 'best_epoch=', report['best_epoch'], flush=True)


if __name__ == '__main__':
    main()
