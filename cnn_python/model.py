
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType

CLASSES = ('circulo', 'cruz', 'triangulo')
SHAPES = {'FILTER1': 9, 'FILTER2': 9, 'W1': 784, 'B1': 8, 'W2': 24, 'B2': 3}
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / 'model' / 'weights' / 'weights.txt'


def finite(value):
    if not math.isfinite(value):
        raise ValueError('nonfinite_result')
    return value


def vector(values, count, normalized=False):
    if not isinstance(count, int) or not 0 < count <= 1_000_000:
        raise ValueError('array_size')
    result = tuple(float(value) for value in values)
    if len(result) != count or not all(math.isfinite(value) for value in result):
        raise ValueError('array_length_or_nonfinite')
    if normalized and not all(0 <= value <= 1 for value in result):
        raise ValueError('pixels_must_be_in_0_1')
    return result


def validate_weights(weights):
    if set(weights) != set(SHAPES):
        raise ValueError('weight_sections')
    return {key: vector(weights[key], count) for key, count in SHAPES.items()}


def load_weights(path=None):
    lines = [line.strip() for line in Path(path if path is not None else DEFAULT_WEIGHTS)
             .read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(lines) != 2 * len(SHAPES):
        raise ValueError('weight_sections')
    weights = {}
    for i in range(0, len(lines), 2):
        key = lines[i]
        if key not in SHAPES or key in weights:
            raise ValueError('unknown_or_duplicate_weight_section')
        weights[key] = vector(lines[i+1].split(','), SHAPES[key])
    return validate_weights(weights)


def save_weights(weights, path):
    weights = validate_weights(weights)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(key + '\n' + ','.join(format(x, '.17g') for x in weights[key]) + '\n'
                   for key in SHAPES)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as output:
        output.write(text)
    temporary.replace(path)


def convolve_valid(image, rows, cols, kernel, kr, kc):
    if (any(type(n) is not int or n < 1 for n in (rows, cols, kr, kc))
            or kr > rows or kc > cols):
        raise ValueError('dims')
    image, kernel = vector(image, rows*cols), vector(kernel, kr*kc)
    out_rows, out_cols = rows-kr+1, cols-kc+1
    result = []
    for r in range(out_rows):
        for c in range(out_cols):
            total = 0.0
            for i in range(kr):
                for j in range(kc):
                    total += image[(r+i)*cols+c+j] * kernel[i*kc+j]
            result.append(finite(total))
    return result, out_rows, out_cols


def relu(values):
    return [max(0.0, finite(float(value))) for value in values]


def pool_with_indices(matrix, rows, cols):
    if any(type(n) is not int or n < 2 for n in (rows, cols)):
        raise ValueError('pool_dims')
    matrix = vector(matrix, rows*cols)
    result, indices = [], []
    for r in range(0, rows-1, 2):
        for c in range(0, cols-1, 2):
            positions = (r*cols+c, r*cols+c+1, (r+1)*cols+c, (r+1)*cols+c+1)
            winner = max(positions, key=matrix.__getitem__)
            result.append(matrix[winner])
            indices.append(winner)
    return result, rows//2, cols//2, indices


def max_pooling(matrix, rows, cols):
    values, out_rows, out_cols, _ = pool_with_indices(matrix, rows, cols)
    return values, out_rows, out_cols


def flatten(channels, rows=7, cols=7):
    if not channels or any(type(n) is not int or n < 1 for n in (rows, cols)):
        raise ValueError('flatten_dims')
    return [value for channel in channels for value in vector(channel, rows*cols)]


def softmax(logits):
    logits = tuple(float(value) for value in logits)
    if not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError('invalid_logits')
    maximum = max(logits)
    exps = [math.exp(value-maximum) for value in logits]
    total = sum(exps)
    return [value/total for value in exps]


def _dense(x, w1, b1, w2, b2):
    ni, nh, no = len(x), len(b1), len(b2)
    prehidden = [finite(sum(w1[i*ni+j]*x[j] for j in range(ni)) + b1[i]) for i in range(nh)]
    hidden = relu(prehidden)
    logits = [finite(sum(w2[i*nh+j]*hidden[j] for j in range(nh)) + b2[i]) for i in range(no)]
    return prehidden, hidden, logits, softmax(logits)


def mlp(x, w1, b1, w2, b2):
    ni, nh, no = len(x), len(b1), len(b2)
    x, w1, b1 = vector(x, ni), vector(w1, ni*nh), vector(b1, nh)
    w2, b2 = vector(w2, nh*no), vector(b2, no)
    return _dense(x, w1, b1, w2, b2)[3]


def _forward(image, weights):
    # Ruta especializada 16x16/3x3: mismas nueve sumas, sin cambiar el calculo.
    maps, pooled, winners = [], [], []
    for key in ('FILTER1', 'FILTER2'):
        k = weights[key]
        conv = []
        for r in range(14):
            for c in range(14):
                p = r*16+c
                conv.append(finite(image[p]*k[0] + image[p+1]*k[1] + image[p+2]*k[2]
                                   + image[p+16]*k[3] + image[p+17]*k[4] + image[p+18]*k[5]
                                   + image[p+32]*k[6] + image[p+33]*k[7] + image[p+34]*k[8]))
        values, _, _, indices = pool_with_indices(relu(conv), 14, 14)
        maps.append(conv)
        pooled.append(values)
        winners.append(indices)
    flat = flatten(pooled)
    prehidden, hidden, logits, probabilities = _dense(flat, weights['W1'], weights['B1'],
                                                    weights['W2'], weights['B2'])
    return {'convolution': maps, 'pooled': pooled, 'winners': winners, 'flat': flat,
            'prehidden': prehidden, 'hidden': hidden, 'logits': logits,
            'probabilities': probabilities}


@dataclass(frozen=True)
class Prediction:
    index: int
    label: str
    probabilities: tuple


class CNN:
    def __init__(self, weights):
        # El modelo de inferencia es inmutable; nunca publica pesos parciales.
        self.weights = MappingProxyType(validate_weights(weights))

    @classmethod
    def load(cls, path=None):
        return cls(load_weights(path))

    def trace(self, pixels):
        return _forward(vector(pixels, 256, normalized=True), self.weights)

    def predict(self, pixels):
        probabilities = tuple(self.trace(pixels)['probabilities'])
        index = max(range(3), key=probabilities.__getitem__)
        return Prediction(index, CLASSES[index], probabilities)


def loss_and_gradients(weights, image, label):
    """Retropropagacion completa: entropia cruzada, MLP, max pooling y filtros.

    Funcion interna de entrenamiento: pesos/imagen validados al cargar el dataset.
    En empates del max pooling se usa la primera posicion; ReLU'(0) = 0.
    """
    if type(label) is not int or not 0 <= label < 3:
        raise ValueError('class_index')
    state = _forward(image, weights)
    logits = state['logits']
    maximum = max(logits)
    loss = finite(maximum + math.log(sum(math.exp(x-maximum) for x in logits)) - logits[label])
    dz = state['probabilities'][:]
    dz[label] -= 1
    gradients = {key: [0.0]*count for key, count in SHAPES.items()}
    gradients['B2'] = dz
    hidden, flat = state['hidden'], state['flat']
    dh = [sum(weights['W2'][i*8+j]*dz[i] for i in range(3))
          if state['prehidden'][j] > 0 else 0.0 for j in range(8)]
    gradients['B1'] = dh
    for i in range(3):
        for j in range(8):
            gradients['W2'][i*8+j] = dz[i]*hidden[j]
    for i in range(8):
        for j in range(98):
            gradients['W1'][i*98+j] = dh[i]*flat[j]
    dflat = [sum(weights['W1'][i*98+j]*dh[i] for i in range(8)) for j in range(98)]
    for channel, key in enumerate(('FILTER1', 'FILTER2')):
        for j, position in enumerate(state['winners'][channel]):
            if state['convolution'][channel][position] <= 0:
                continue
            delta = dflat[channel*49+j]
            r, c = divmod(position, 14)
            for i in range(3):
                for k in range(3):
                    gradients[key][i*3+k] += delta*image[(r+i)*16+c+k]
    return loss, gradients
