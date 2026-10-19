# Plan mínimo de experimentos

## Escenarios del enunciado
1. Secuencial: un proceso, un hilo de cálculo, CNN local.
2. Paralelo: una máquina, varios núcleos físicos, múltiples hilos.
3. Distribuido: M máquinas, un núcleo (verificar asignación real) y un hilo de cómputo por nodo.
4. Paralelo distribuido: M máquinas y varios núcleos/hilos por nodo.

## Buen control experimental
- Exactamente mismo conjunto de imágenes, pesos, cálculos y salidas en todos los casos.
- Solicitudes 100/300/500/1000 con semillas y delays aleatorios reproducibles.
- Latencia por solicitud, P50/P95, throughput, tiempo total, fallos y speedup.
- 3 o más repeticiones, warm-up separado si se usa JIT, red/equipos/cores documentados.
- No inventar resultados ni confundir rendimiento de varias imágenes con paralelismo dentro de una imagen.
