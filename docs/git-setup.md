# Preparar el repositorio y trabajar por ramas

Esta carpeta se entrega como ZIP de arranque (sin `.git`). El integrante 1 crea primero un repositorio **vacío** de GitHub sin README automático, y todos trabajan desde allí.

## Inicialización (solo integrante 1)

```bash
cd OpenAICubes
git init -b main
git add .
git commit -m "chore: initial project structure and TCP micro-chunk smoke tests"
git remote add origin https://github.com/ORGANIZACION/OpenAICubes.git
git push -u origin main
git switch -c develop
git push -u origin develop
```

Opcional: proteger `main` y exigir PR antes de merge.

## Comienzo de cada integrante (clonar, no copiar carpetas separadas)

```bash
git clone https://github.com/ORGANIZACION/OpenAICubes.git
cd OpenAICubes
git fetch origin
git switch -c feat/java-server origin/develop
```

Ramas equivalentes: `feat/cnn-python`, `feat/csharp-benchmark`. Cada integrante usa **solo su propia rama**.

## Ciclo de implementación

```bash
git status
git add <archivos-propios>
git commit -m "feat: describe el cambio"
git push -u origin <rama>
```

Crear **Pull Request con destino `develop`**. Revisar el código y ejecutar las pruebas antes de fusionar.

## Nuevas tareas tras integrar una rama

```bash
git fetch origin
git switch develop
git pull --ff-only origin develop
git switch -c feat/nombre-siguiente-cambio
```

## Final del día 3

- Ejecutar smoke tests y demostración en `develop`.
- Abrir PR `develop` → `main`; fusionarlo cuando esté validado.
- Etiquetar la versión probada:

```bash
git switch main
git pull --ff-only origin main
git tag parcial-v1.0
git push origin parcial-v1.0
```

- Generar ZIP de fuentes y documentación **sin** `.git`, `build/`, `bin/`, `obj/`, cachés y logs.
