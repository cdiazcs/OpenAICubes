# Plan de trabajo: 3 personas / 3 días

## Ramas
- `main`: versión probada, final.
- `develop`: integración y smoke tests.
- `feat/java-server`: integrante 1 — coordinador, frontend, worker Java.
- `feat/cnn-python`: integrante 2 — CNN de referencia, pesos y worker Python.
- `feat/csharp-benchmark`: integrante 3 — worker C#, clúster y medición.

## Día 1
1. Acordar y congelar `docs/protocol.md` y `docs/model-spec.md`.
2. Compilar/probar `PING`, `CONV`, `MLP` de Java y Python; completar validación C#.
3. Terminar primer flujo local de CNN secuencial en Java; guardar datos/pesos reproducibles.
4. Abrir PRs pequeños a `develop` y ejecutar prueba integrada.

## Día 2
1. Conectar CNN a sockets, división real en micro chunks y ensamblado correcto.
2. Implementar frontend Swing, concurrencia y despliegue de tres máquinas.
3. Demostrar resultados iguales entre base secuencial y sistema distribuido.
4. Dejar funcionando extremo a extremo antes de dormir.

## Día 3
1. Evaluar cuatro escenarios del enunciado bajo condiciones documentadas, con cientos de solicitudes y delay aleatorio.
2. Registrar latencia, throughput, fallos, speedup y gráficas usando herramientas permitidas.
3. Informe + presentación PDF + diagrama de secuencia + guion.
4. Congelar, probar clon limpio, fusionar `develop` -> `main`, etiquetar versión y generar ZIP sin `.git`.

## Norma de PR
- Una revisión cruzada antes de fusionar.
- No romper las pruebas actuales; cambiar protocolo solo por acuerdo.
- Nada de trabajo directo en `main`.
