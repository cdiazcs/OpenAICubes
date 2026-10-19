# Protocolo Open AI Cubes (v0.1)

## Transporte
- **TCP sin middleware**; un mensaje = 4 bytes de longitud sin signo, big-endian, seguidos de exactamente esa cantidad de bytes UTF-8.
- La longitud máxima aceptada por estos prototipos es 2,000,000 bytes (límite por conexión/mensaje para prevenir abuso).
- Una conexión permite múltiples mensajes. Se envía un mensaje de respuesta por solicitud.
- Los campos van separados por `|`; ningún campo de la versión 0.1 puede contener ese carácter.
- Un vector se serializa como `0.1,0.2,-3.5`, en orden de filas, con punto decimal. Las dimensiones viajan explícitas.
- Mensajes sin normalizar: el receptor no debe inferir dimensiones. Los mensajes con errores generan `ERROR|id|descripcion`.
- Los números se interpretan como `double` IEEE-754 en los tres lenguajes; tolerancia sugerida para comparaciones: 1e-8.

## Tipos actuales

`PING|requestId` → `PONG|requestId|language`

`CONV|requestId|rows|cols|kernelRows|kernelCols|imageCSV|kernelCSV`

`RESULT|requestId|outRows|outCols|valuesCSV`

- Convolución válida sin padding, stride = 1. Se implementa correlación cruzada (convención utilizada frecuentemente en CNN). No hay bias, ReLU ni pooling todavía.
- `outRows = rows - kernelRows + 1`, `outCols = cols - kernelCols + 1`.

`MLP|requestId|nInputs|nHidden|nOutputs|inputCSV|weights1CSV|bias1CSV|weights2CSV|bias2CSV`

`RESULT|requestId|1|nOutputs|probabilitiesCSV`

- `weights1` tiene `nHidden*nInputs` valores por filas (un renglón por neurona).
- `weights2` tiene `nOutputs*nHidden` valores por filas.
- La red calcula `h = ReLU(W1*x + b1)`, `logits = W2*h + b2`, `y = softmax(logits)`.
- Es una operación válida de MLP pero en esta versión **no representa todavía la CNN completa**: faltan extracción de características completa, pooling, pesos entrenados y preparación de imagen.

## Ejemplos verificables

Entrada CONV:
`CONV|t1|3|3|2|2|1,2,3,4,5,6,7,8,9|1,0,0,-1`
Salida: `RESULT|t1|2|2|-4,-4,-4,-4`

Entrada MLP:
`MLP|t2|2|2|2|1,2|1,0,0,1|0,0|1,0,0,1|0,0`
Salida aproximada: `RESULT|t2|1|2|0.2689414213699951,0.7310585786300049`

## Próxima versión (consensuar antes de modificar)
- `IDENTIFY`: envío de imagen + preprocesamiento y predicción real.
- `HELLO`/`READY`: registro/capacidades de nodo y balanceo por capacidad.
- `JOB`/`PART`: fragmentación de conv/MLP en micro chunks repartidos entre múltiples workers y reunión determinista.
- Errores recuperables, identificadores y reintentos idempotentes.

**Regla de equipo:** cualquier cambio incompatible exige actualizar este documento y las 3 implementaciones mediante revisión de PR.
