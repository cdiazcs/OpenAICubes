
import csv
import argparse
import hashlib
import math
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape


def load(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('CSV sin filas: ' + str(path))
    missing = {'modo', 'latencia_ms', 'ok'} - set(rows[0])
    if missing:
        raise ValueError('Faltan columnas ' + ', '.join(sorted(missing)) + ' en ' + str(path))
    ms = []
    for row in rows:
        if row['ok'].strip() == '1':
            ms.append(float(row['latencia_ms']))
    ms.sort()
    n = len(rows)
    good = len(ms)
    p95 = ms[max(0, math.ceil(good * 0.95) - 1)] if good else 0.0
    return {
        'label': path.stem,
        'requests': n,
        'ok': good,
        'errors': n - good,
        'mean': sum(ms) / good if good else 0.0,
        'p95': p95,
        'success_pct': 100.0 * good / n,
    }


def svg_base(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            '<rect width="100%" height="100%" fill="white"/>\n'
            '<g font-family="Arial, sans-serif" fill="#273142">\n'
            + '\n'.join(body) + '\n</g>\n</svg>\n')


def chart_latencies(entries):
    w, h = max(720, len(entries)*154 + 170), 465
    left, right, top, bottom = 85, 32, 84, 100
    xwidth, yheight = w - left - right, h - top - bottom
    ymax = max(1.0, max(max(a['mean'], a['p95']) for a in entries) * 1.20)
    body = [f'<text x="{w/2}" y="34" font-size="20" font-weight="bold" text-anchor="middle">Latencia por carga: worker C#</text>',
            f'<text x="{w/2}" y="55" font-size="12" fill="#666" text-anchor="middle">Medidas en ms de solicitudes correctas</text>']
    for tick in range(5):
        v = ymax * tick/4
        y = top + yheight - yheight*tick/4
        body.extend([f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                     f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="11">{v:.2f}</text>'])
    groupw = xwidth / len(entries)
    barw = min(36, groupw/3.2)
    for i, row in enumerate(entries):
        center = left + (i+0.5)*groupw
        for off, key, color in [(-barw-3, 'mean', '#3b82f6'), (3, 'p95', '#ef9f28')]:
            val = row[key]
            barh = yheight * val/ymax
            x = center+off
            y = top + yheight-barh
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{barw:.1f}" height="{barh:.1f}" fill="{color}" rx="3"/>')
            body.append(f'<text x="{x+barw/2:.1f}" y="{max(top-3,y-7):.1f}" font-size="10" text-anchor="middle">{val:.3f}</text>')
        body.append(f'<text x="{center:.1f}" y="{top+yheight+24}" font-size="12" text-anchor="middle">{row["requests"]} req</text>')
    body.extend([f'<rect x="{left}" y="{h-49}" width="13" height="13" fill="#3b82f6"/>',
                 f'<text x="{left+19}" y="{h-38}" font-size="12">Media</text>',
                 f'<rect x="{left+115}" y="{h-49}" width="13" height="13" fill="#ef9f28"/>',
                 f'<text x="{left+134}" y="{h-38}" font-size="12">P95</text>'])
    return svg_base(w,h,body)


def chart_errors(entries):
    w, h = max(720, len(entries)*154 + 170), 435
    left, right, top, bottom = 80, 25, 77, 90
    xwidth, yheight = w-left-right, h-top-bottom
    ymax = max(1, math.ceil(max(e['errors'] for e in entries)*1.2))
    body = [f'<text x="{w/2}" y="34" font-size="20" font-weight="bold" text-anchor="middle">Errores por carga</text>',
            f'<text x="{w/2}" y="55" font-size="12" fill="#666" text-anchor="middle">Solicitudes no correctas / no enviadas</text>']
    for tick in range(5):
        v = ymax*tick/4
        y = top+yheight-yheight*tick/4
        body.extend([f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                     f'<text x="{left-12}" y="{y+4:.1f}" font-size="11" text-anchor="end">{v:g}</text>'])
    groupw=xwidth/len(entries)
    for i,e in enumerate(entries):
        cx=left+(i+0.5)*groupw
        height=yheight*e['errors']/ymax
        body.append(f'<rect x="{cx-22:.1f}" y="{top+yheight-height:.1f}" width="44" height="{height:.1f}" fill="#c65c67" rx="3"/>')
        body.append(f'<text x="{cx:.1f}" y="{max(top-4,top+yheight-height-9):.1f}" font-size="12" text-anchor="middle">{e["errors"]}</text>')
        body.append(f'<text x="{cx:.1f}" y="{top+yheight+24}" font-size="12" text-anchor="middle">{e["requests"]} req</text>')
    return svg_base(w,h,body)


def ordering(path):
    m = re.search(r'(\d+)$', path.stem)
    return int(m.group(1)) if m else 10**10


def legacy_main():
    indir = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path('benchmarks/results')
    output = Path(sys.argv[2]) if len(sys.argv) >= 3 else indir.parent/'graphs'
    paths = sorted(indir.glob('csharp-*.csv'), key=ordering)
    if not paths:
        raise SystemExit('No se encontraron archivos csharp-*.csv en ' + str(indir))
    data = [load(p) for p in paths]
    output.mkdir(parents=True, exist_ok=True)
    with (output/'resumen.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f)
        w.writerow(['prueba','solicitudes','correctas','errores','exito_pct','latencia_media_ms','p95_ms'])
        for row in data:
            w.writerow([row['label'], row['requests'], row['ok'], row['errors'],
                        f'{row["success_pct"]:.2f}', f'{row["mean"]:.6f}', f'{row["p95"]:.6f}'])
    (output/'latencias.svg').write_text(chart_latencies(data), encoding='utf-8')
    (output/'errores.svg').write_text(chart_errors(data), encoding='utf-8')
    print('Graficas y resumen creados en:', output.resolve())
    print('Carga | Correctas | Errores | Media (ms) | P95 (ms)')
    for row in data:
        print(f'{row["requests"]:>5} | {row["ok"]:>9} | {row["errors"]:>7} | {row["mean"]:>10.3f} | {row["p95"]:>8.3f}')
    print('NOTA: throughput requiere duracion total; usa la salida de Benchmark.java.')


def properties(path):
    values = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith(('#', '!')):
            continue
        key, separator, value = line.partition('=')
        if separator:
            values[key] = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), value).replace('\\:', ':').replace('\\=', '=').replace('\\\\', '\\')
    return values


def measured(path, allow_localhost=False):
    summary = path.with_name(path.name+'.summary.properties')
    if not summary.exists():
        return None
    data = properties(summary)
    if data.get('schema') != 'openaicubes-benchmark-6' or data.get('operation') != 'IDENTIFY':
        return None
    if data.get('scenario') not in ('LOCAL_SEQUENTIAL', 'LOCAL_PARALLEL', 'DISTRIBUTED', 'PARALLEL_DISTRIBUTED'):
        return None
    if data.get('configuration_valid') != 'true' or (data.get('verification') == 'localhost_emulation' and not allow_localhost):
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != data.get('requests.csv.sha256'):
        raise ValueError('CSV checksum: '+str(path))
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row['requestId'] for row in rows}) != len(rows):
        raise ValueError('request IDs: '+str(path))
    latencies, causes = [], {}
    for row in rows:
        latency = float(row['latencia_ms'])
        if not math.isfinite(latency) or latency <= 0 or row['ok'] not in ('0', '1'):
            raise ValueError('invalid measurement: '+str(path))
        if row['ok'] == '1':
            probabilities = [float(row['p'+str(i)]) for i in range(3)]
            if row['error_causa'] or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities)-1) > 1e-8:
                raise ValueError('invalid success: '+str(path))
            latencies.append(latency)
        else:
            cause = row['error_causa']
            if not cause or 'errors.'+cause not in data:
                raise ValueError('invalid error cause: '+str(path))
            causes[cause] = causes.get(cause, 0)+1
    latencies.sort()
    checks = {'samples': len(rows), 'success': len(latencies), 'errors': len(rows)-len(latencies)}
    checks.update({'errors.'+key[7:]: causes.get(key[7:], 0) for key in data if key.startswith('errors.')})
    total = float(data['total.s'])
    if not math.isfinite(total) or total <= 0:
        raise ValueError('invalid duration')
    checks['throughput.rps'] = len(latencies)/total
    if latencies:
        checks['mean.ms'] = sum(latencies)/len(latencies)
        checks['p95.ms'] = latencies[math.ceil(.95*len(latencies))-1]
    for key, value in checks.items():
        if not math.isclose(float(data[key]), value, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError('metric mismatch '+key+': '+str(path))
    data.update(file=str(path), label=path.stem, causes=causes)
    return data


def measured_chart(entries, metrics, title, unit):
    width, height = max(800, len(entries)*155+130), 490
    left, right, top, base = 85, 25, 75, 340
    present = [(entry, key, color) for entry in entries for key, color in metrics if entry.get(key, '') != '']
    maximum = max([float(entry[key]) for entry, key, _ in present]+[1e-9])*1.15
    fingerprints = {entry['load.sha256'] for entry in entries}
    scope = 'carga '+entries[0]['load.sha256'][:12] if len(fingerprints) == 1 else str(len(fingerprints))+' cargas medidas por clientes'
    body = [f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20">{escape(title)}</text>',
            f'<text x="{width/2}" y="50" text-anchor="middle" font-size="11">IDENTIFY medido; {escape(unit)}; {escape(scope)}</text>']
    for tick in range(6):
        y = base-(base-top)*tick/5
        body.extend([f'<line x1="{left}" x2="{width-right}" y1="{y}" y2="{y}" stroke="#ddd"/>',
                     f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="10">{maximum*tick/5:.4g}</text>'])
    step = (width-left-right)/len(entries)
    for index, entry in enumerate(entries):
        center = left+(index+.5)*step
        for at, (key, color) in enumerate(metrics):
            if entry.get(key, '') == '':
                continue
            value = float(entry[key]); bar = (base-top)*value/maximum
            x = center+(at-len(metrics)/2)*28
            body.extend([f'<rect x="{x}" y="{base-bar}" width="24" height="{bar}" fill="{color}"/>',
                         f'<text x="{x+12}" y="{base-bar-5}" text-anchor="middle" font-size="10">{value:.4g}</text>'])
        label = escape(entry['label']+' c='+entry['clients'])
        body.append(f'<text transform="translate({center-8},{base+15}) rotate(30)" font-size="10">{label}</text>')
    for at, (key, color) in enumerate(metrics):
        body.append(f'<text x="{left+at*240}" y="{height-12}" fill="{color}" font-size="12">{escape(key)}</text>')
    return svg_base(width, height, body)


def generate(indir, output, allow_localhost=False):
    entries = []
    for path in sorted(indir.glob('*.csv')):
        entry = measured(path, allow_localhost)
        if entry:
            entries.append(entry)
    if not entries:
        raise ValueError('No hay mediciones IDENTIFY validas; CSV historicos CONV se excluyen')
    output.mkdir(parents=True, exist_ok=True)
    columns = ['label', 'scenario', 'verification', 'clients', 'samples', 'success', 'errors', 'total.s', 'service.s', 'mean.ms', 'p95.ms', 'throughput.rps', 'pure.cpu.ms', 'speedup.total', 'speedup.service', 'load.sha256', 'weights.sha256']
    with (output/'resumen.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(entries)
    groups = {}
    for entry in entries:
        groups.setdefault(entry['load.sha256'], []).append(entry)
    for fingerprint, group in groups.items():
        destination = output/('load-'+fingerprint[:12])
        destination.mkdir(exist_ok=True)
        plots = [('tiempo', [('total.s', '#4b81c7')], 'Tiempo total', 's'),
                 ('latencias', [('mean.ms', '#4b81c7'), ('p95.ms', '#ed9b34')], 'Latencia media y P95', 'ms'),
                 ('throughput', [('throughput.rps', '#3b9679')], 'Throughput correcto', 'solicitudes/s'),
                 ('speedup', [('speedup.total', '#7e64bb'), ('speedup.service', '#bb647f')], 'Speedup frente al secuencial equivalente', 'razon'),
                 ('errores', [('errors', '#c65c67')], 'Errores y clientes concurrentes', 'solicitudes')]
        for name, metrics, title, unit in plots:
            if any(entry.get(key, '') != '' for entry in group for key, _ in metrics):
                (destination/(name+'.svg')).write_text(measured_chart(group, metrics, title, unit), encoding='utf-8')
    (output/'errores-clientes.svg').write_text(measured_chart(entries, [('errors', '#c65c67')], 'Errores por configuracion y clientes', 'solicitudes'), encoding='utf-8')
    print('Mediciones IDENTIFY:', len(entries), 'grupos de carga:', len(groups), 'salida:', output)
    return entries


def main():
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy'); legacy_main(); return
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, nargs='?', default=Path('build/stage6/measurements'))
    parser.add_argument('output', type=Path, nargs='?', default=Path('build/stage6/graphs'))
    parser.add_argument('--allow-localhost', action='store_true')
    args = parser.parse_args()
    generate(args.input, args.output, args.allow_localhost)


if __name__ == '__main__':
    main()
