# Open AI Cubes

**Proyecto del curso de Programación Concurrente y Distribuida — CC4P1**

#Integrantes
- Maurizio Berdiales
- Enoc Rosales
- Cristian Diaz

## ¿De qué trata nuestro proyecto?

Open AI Cubes es un proyecto que desarrollamos para el curso de Programación Concurrente y Distribuida.

Nuestro objetivo fue implementar un sistema capaz de identificar imágenes utilizando una **red neuronal convolucional (CNN)** y distribuir las operaciones de procesamiento entre diferentes nodos.

Para esto utilizamos **Java, Python y C#**, conectando los componentes mediante sockets TCP. También aplicamos conceptos de concurrencia, paralelismo, comunicación entre procesos y coordinación de tareas.

Una de las principales restricciones del proyecto fue trabajar únicamente con las bibliotecas estándar de cada lenguaje, sin utilizar frameworks ni librerías externas de inteligencia artificial.

## Funcionamiento del sistema

El sistema está organizado en varios componentes que trabajan de manera conjunta.

El usuario puede seleccionar una imagen desde una interfaz gráfica desarrollada en Java Swing. Esta imagen se envía al servidor central Java, que se encarga de coordinar las operaciones de la CNN.

Dependiendo del modo seleccionado, el procesamiento puede realizarse secuencialmente o mediante micro chunks distribuidos entre los workers disponibles.

Los nodos realizan los cálculos correspondientes y devuelven sus resultados al servidor, que finalmente obtiene la clasificación de la imagen y la muestra en la interfaz.

El flujo general es:

```text
              Usuario
                 |
                 v
        Frontend Java Swing
                 |
                 v
        Servidor Central Java
                 |
          Sockets TCP
                 |
        +--------+--------+
        |        |        |
        v        v        v
     Java     Python     C#
     Worker   Worker    Worker
        |        |        |
        +--------+--------+
                 |
                 v
        Resultado de la CNN
                 |
                 v
        Clasificación final
```

## Implementación de la CNN

Desarrollamos una red neuronal convolucional desde cero utilizando Python y sus bibliotecas estándar.

El modelo trabaja con imágenes de **16 × 16 píxeles** y realiza las siguientes operaciones:

1. Lectura y normalización de la imagen.
2. Aplicación de dos filtros convolucionales de 3 × 3.
3. Función de activación ReLU.
4. Max Pooling.
5. Conversión de los resultados a un vector de 98 características.
6. Procesamiento mediante un MLP de 98 → 8 → 3 neuronas.
7. Aplicación de Softmax para obtener las probabilidades de clasificación.

Para nuestro proyecto trabajamos con tres figuras geométricas:

- Círculo
- Cruz
- Triángulo

Elegimos estas categorías para poder implementar, entrenar y evaluar la CNN, concentrándonos también en la parte concurrente y distribuida del curso.

## Entrenamiento y resultados

Para entrenar el modelo generamos imágenes sintéticas con diferentes posiciones, tamaños y variaciones de las figuras.

El conjunto de datos se dividió de la siguiente manera:

| Conjunto | Imágenes | Uso |
|---|---:|---|
| Entrenamiento | 720 | Entrenar los parámetros de la CNN |
| Validación | 180 | Seleccionar el mejor modelo |
| Prueba | 300 | Evaluar las clasificaciones |
| **Total** | **1200** | |

La CNN cuenta con **837 parámetros entrenables**.

Utilizamos semillas diferentes para generar los conjuntos de datos y hacer que el entrenamiento fuera reproducible.

En las pruebas del modelo obtuvimos **300 clasificaciones correctas de 300 imágenes evaluadas**, alcanzando una precisión del 100 % en nuestro conjunto de prueba sintético.

Este resultado corresponde únicamente a las figuras geométricas utilizadas en el proyecto. No significa que la CNN pueda reconocer cualquier imagen u objeto del mundo real.

Los pesos entrenados se encuentran almacenados en `model/weights/`.

## Programación concurrente y distribuida

Una de las partes principales del proyecto fue implementar la comunicación y el procesamiento entre nodos.

Para esto desarrollamos un protocolo propio basado en sockets TCP, permitiendo intercambiar mensajes y resultados entre Java, Python y C#.

Las operaciones implementadas incluyen:

- `PING`: comprobar la comunicación con los nodos.
- `CONV`: realizar operaciones de convolución.
- `MLP`: ejecutar cálculos del perceptrón multicapa.
- `IDENTIFY`: procesar una imagen y obtener su clasificación.

También implementamos un coordinador en Java para distribuir operaciones independientes de la CNN.

El coordinador puede dividir cálculos convolucionales y operaciones de las capas densas, enviar tareas a distintos workers y reunir los resultados para continuar el procesamiento.

De esta manera aplicamos conceptos del curso como threads, ejecución concurrente, paralelismo, sockets y paso de mensajes.

## Interfaz gráfica

Desarrollamos una interfaz de escritorio utilizando **Java Swing**, desde la cual podemos interactuar con el sistema.

La aplicación permite seleccionar imágenes, visualizarlas y enviarlas al servidor para obtener una clasificación.

También muestra la clase predicha, las probabilidades y el tiempo de respuesta de la solicitud.

La interfaz incluye dos modalidades de inferencia:

- **SEQUENTIAL:** ejecuta la CNN de manera secuencial.
- **DISTRIBUTED:** utiliza el coordinador para repartir cálculos entre los nodos configurados.

Esto nos permite observar el comportamiento de la misma CNN utilizando diferentes estrategias de procesamiento.

## Pruebas de comunicación y rendimiento

Implementamos pruebas automáticas para verificar el funcionamiento de los componentes.

Entre ellas se encuentran:

- Validación de operaciones matemáticas.
- Comunicación TCP entre procesos.
- Compatibilidad de resultados entre lenguajes.
- Manejo de mensajes incorrectos.
- Atención concurrente de solicitudes.
- Comparación numérica de inferencias secuenciales y distribuidas.

También desarrollamos un generador de carga que permite enviar cientos de solicitudes, controlar la cantidad de clientes concurrentes y utilizar retrasos aleatorios.

Durante las pruebas iniciales registramos latencia, throughput y errores, almacenando los resultados en archivos CSV para generar gráficas.

Estas mediciones iniciales se realizaron sobre operaciones de convolución. La comparación definitiva de la CNN completa en los cuatro escenarios del parcial debe respaldarse con los resultados experimentales correspondientes.

## Tecnologías utilizadas

| Tecnología | Uso en el proyecto |
|---|---|
| Java | Servidor central, frontend, coordinación y concurrencia |
| Python | CNN, entrenamiento, evaluación y worker |
| C# | Worker de procesamiento |
| TCP Sockets | Comunicación entre nodos |
| Java Swing | Interfaz gráfica |
| Git y GitHub | Control de versiones |
| VS Code | Desarrollo y pruebas |

Todo el procesamiento se implementó con las bibliotecas estándar de los lenguajes utilizados.

## Estructura del proyecto

```text
OpenAICubes/
│
├── benchmarks/
│
├── clients-java/
│
├── cnn-java/
│
├── cnn_python/
│
├── docs/
│
├── frontend-java/
│
├── model/
│   ├── weights/
│   ├── data/
│   ├── examples/
│   ├── training/
│   └── evaluation/
│
├── server-java/
│
├── tests/
│
├── workers/
│   ├── java/
│   ├── python/
│   └── csharp/
│
└── README.md
```

Organizamos el proyecto por módulos para facilitar las pruebas, el mantenimiento del código y la integración de las diferentes implementaciones.

## Requisitos para ejecutar el proyecto

Es necesario tener instalados:

- Java JDK 8 o superior.
- Python 3.8 o superior.
- .NET SDK 8.

No se necesitan frameworks externos ni librerías de inteligencia artificial.

## Ejecución y pruebas

Los siguientes comandos se ejecutan desde la carpeta principal del proyecto.

### Probar la CNN localmente

```powershell
python -m cnn_python.predict model/examples/circulo-0.png
python -m cnn_python.predict model/examples/cruz-0.png
python -m cnn_python.predict model/examples/triangulo-0.png
```

### Ejecutar las pruebas automáticas

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### Volver a entrenar y evaluar la CNN

```powershell
python -m cnn_python.dataset
python -m cnn_python.train
python -m cnn_python.evaluate
```

### Probar la comunicación TCP

```powershell
python tests/test_protocol.py
```

Estas pruebas permiten comprobar el funcionamiento matemático y la comunicación de los componentes.

Para ejecutar el sistema completo se deben iniciar los workers, el servidor central Java y posteriormente la interfaz Swing, utilizando las configuraciones de IP y puerto correspondientes.

## Organización del equipo

El proyecto fue desarrollado por tres integrantes, dividiendo las principales responsabilidades de la siguiente manera:

| Integrante | Responsabilidad |
|---|---|
| Integrante 1 | Servidor Java, sockets e interfaz Swing |
| Integrante 2 | CNN, entrenamiento, pesos y worker Python |
| Integrante 3 | Worker C#, benchmarks y despliegue distribuido |

Utilizamos GitHub para organizar el trabajo, desarrollar módulos de manera independiente e integrar los avances.

La rama `develop` se utilizó para integrar y probar las implementaciones. La rama `main` se reservó para la versión final del proyecto.

## Estado final del desarrollo

Hemos completado la implementación principal de Open AI Cubes, incluyendo la CNN secuencial, el procesamiento de imágenes, la comunicación mediante sockets TCP, los workers en tres lenguajes, el coordinador distribuido y el cliente de escritorio Java Swing.

Con este proyecto logramos aplicar de manera práctica varios conceptos desarrollados durante el curso de Programación Concurrente y Distribuida.

Las pruebas locales nos permitieron validar los cálculos y la comunicación entre componentes. La ejecución física del clúster en diferentes computadoras y la comparación de los cuatro escenarios de rendimiento deben verificarse con sus respectivas evidencias experimentales.

## Consideraciones

El proyecto fue desarrollado con fines académicos. El protocolo TCP está pensado para ejecutarse dentro de una red local de confianza y no incorpora mecanismos de autenticación o cifrado.

Por otro lado, la CNN fue entrenada con imágenes sintéticas de figuras geométricas, por lo que su funcionamiento está orientado a las categorías utilizadas durante el desarrollo.

---

**Open AI Cubes — CC4P1 — 2026-II**

*Proyecto académico de Programación Concurrente y Distribuida.*