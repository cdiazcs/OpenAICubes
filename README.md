# Open AI Cubes — estructura inicial (v0.1)

Proyecto para el parcial CC4P1: concurrencia, paralelismo y sistemas distribuidos con **solo SDK/bibliotecas estándar**.

> **Estado actual:** starter de red funcional con micro chunks `CONV` y `MLP`; **NO** es aún la solución completa del parcial. Faltan CNN de imágenes, entrenamiento/pesos, frontend Swing, reparto de un mismo micro chunk entre nodos y benchmarks de 4 escenarios.

## Equipos y ramas

- Integrante 1: `feat/java-server` → servidor Java, frontend Swing y worker Java.
- Integrante 2: `feat/cnn-python` → CNN matemática, dataset/pesos y worker Python.
- Integrante 3: `feat/csharp-benchmark` → worker C#, pruebas de carga y clúster.
- `develop`: integración de PRs; `main`: versión final probada.

Documentos compartidos: `docs/protocol.md`, `docs/model-spec.md`, `docs/architecture.md`, `docs/work-plan.md`, `docs/experiments.md`.

## Estructura

- `server-java/`: servidor central con threads y round-robin.
- `clients-java/`: cliente CLI para pruebas (no sustituye al frontend).
- `frontend-java/`: módulo Swing por desarrollar.
- `cnn-java/`: CNN secuencial de referencia por desarrollar.
- `workers/java/`, `workers/python/`, `workers/csharp/`: workers micro chunks.
- `model/weights/`: pesos de CNN cuando estén validados.
- `tests/`: casos numéricos comparables por protocolo.
- `benchmarks/`: mediciones y gráficas por desarrollar.

## Requisitos técnicos

- **Java 8+** (código usa solo APIs Java 8), `javac` y `java`.
- **Python 3.8+**, biblioteca estándar.
- **.NET SDK 8** para `workers/csharp`, sin NuGet externos.
- Abrir puertos TCP 9000, 9101, 9102, 9103 si se ejecutan en máquinas diferentes; usar direcciones LAN correctas.

## Arranque manual (cuatro terminales)

Desde la raíz del proyecto:

### 1. Compilar Java

```bash
mkdir -p build
javac -d build server-java/CentralServer.java clients-java/ClientCli.java workers/java/WorkerJava.java
```

### 2. Arrancar workers (una terminal por worker)

```bash
java -cp build WorkerJava 9101
```

```bash
python3 workers/python/worker.py 9102
```

```bash
dotnet run --project workers/csharp/Worker.csproj -- 9103
```

### 3. Arrancar servidor central

```bash
java -cp build CentralServer 9000 127.0.0.1:9101 127.0.0.1:9102 127.0.0.1:9103
```

Si alguno de los workers no está disponible, el servidor intentará los demás; para la demostración de **tres lenguajes** deben estar efectivamente activos los tres.

### 4. Ejecutar cliente de prueba

```bash
java -cp build ClientCli 127.0.0.1 9000
```

Se espera: `PONG`, `RESULT|demo1|2|2|-4,-4,-4,-4` y `RESULT|demo2|1|2|0.26894...,0.73105...`.

### 5. Probar igualdad entre nodos

```bash
python3 tests/smoke_test.py 127.0.0.1:9101 127.0.0.1:9102 127.0.0.1:9103 127.0.0.1:9000
```

Al usar máquinas separadas, sustituir las IPs LAN e iniciar allí el worker respectivo. No pasar `localhost` si el worker está en otra computadora.

## Ejemplo de división de trabajo (paso siguiente)

1. Congelar `docs/protocol.md` y `docs/model-spec.md` con la aprobación de todo el equipo.
2. Implementar imagen y CNN de referencia (Conv + ReLU + Pooling + Flatten + MLP) y pesos que clasifiquen de verdad.
3. Cambiar coordinador para distribuir micro chunks dentro del cálculo de una imagen y ensamblar sus resultados.
4. Construir frontend Swing e identificar imágenes.
5. Implementar los cuatro escenarios, mediciones y saturación de cientos de solicitudes.
6. Hacer PRs a `develop`; fusionar a `main` cuando esté validado.

## Seguridad / límites

Protocolo v0.1 experimental pensado para un clúster de laboratorio, sin autenticación y sin cifrado. Ejecutarlo únicamente en red de confianza. Los puertos y el tamaño máximo de mensajes están documentados en `docs/protocol.md`.

## Entrega de Univirtual

Preparar ZIP **limpio**, no subir `.git`, `build/`, `bin/`, `obj/`, `__pycache__/` ni archivos temporales. Sí incluir fuentes (`.java`, `.py`, `.cs`, `.csproj`), pesos y documentación/PDF que sean necesarios para reproducir la entrega. Verificar restricciones exactas con el docente.

## Inicializar GitHub desde este ZIP

Consultar [`docs/git-setup.md`](docs/git-setup.md): indica la creación de `main`, `develop`, ramas individuales y las reglas de PR. Importante: usar **un único repositorio remoto** compartido.

Windows PowerShell: la carpeta de compilación puede crearse con `New-Item -ItemType Directory -Force build`; la ejecución de Java, Python y .NET es equivalente (con `py` si no hay comando `python3`).
