"""Genera resumen CSV y graficas SVG desde los resultados de Benchmark.java.
Solo biblioteca estandar Python; compatible con Python 3.8+.

Uso desde la raiz de OpenAICubes:
    py benchmarks/GraficasBenchmark.py benchmarks/results

Lee csharp-*.csv y escribe benchmarks/graphs/{resumen.csv,latencias.svg,errores.svg}.
IMPORTANTE: los CSV por solicitud no incluyen duracion total del experimento;
para throughput usa la salida real de Benchmark.java, no infieras ese dato.
"""
import csv
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


def main():
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


if __name__ == '__main__':
    main()
