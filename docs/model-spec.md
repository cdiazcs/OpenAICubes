# CNN: contrato matemático inicial

## Meta del parcial
CNN funcional para imágenes, con extracción de características convolucional + ReLU + Pooling + MLP y pesos que produzcan predicciones justificables.

## Arquitectura propuesta (puede variar solo por acuerdo)
- Entrada: imagen gris 16 x 16, normalizada en [0,1].
- Convolución válida: 2 filtros de 3 x 3 -> mapas 14 x 14 (2 canales).
- ReLU: `max(0,x)`.
- Max pooling: ventanas de 2 x 2 stride 2 -> dos mapas 7 x 7.
- Flatten: 98 componentes, orden de canales y filas documentado.
- MLP: 98 -> 8 -> 3, ReLU en hidden, softmax en salida.
- Clases tentativas: circulo / cruz / triangulo **pendientes de confirmación docente**.

## Decisiones pendientes del integrante de CNN
- Conjunto de entrenamiento/prueba reproducible (sin librerías externas).
- Cómo se obtienen los pesos; filtros entrenables o fijos, y qué método lo justifica.
- Guardado/lectura de pesos en archivos de texto compatibles.
- Validación de exactitud y equivalencia numérica secuencial / distribuida.

## Ya incluido en el starter
- Micro chunk `CONV` de 1 canal + 1 kernel, correlación válida.
- Micro chunk `MLP` de dos capas densas, con ReLU y softmax.

**No afirmar que este starter ya clasifica imágenes:** todavía no encadena imagen real, filtros, pooling, pesos válidos ni interfaz.
