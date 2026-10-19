# Arquitectura de Open AI Cubes

```
Frontend Java Swing (pendiente)
          |
     Socket TCP
          |
Servidor Java (CentralServer)
    pool de hilos para clientes
          |
    sockets TCP / protocolo propio
          |
  +-------+--------+
  |       |        |
Worker   Worker   Worker
Java     Python   C#
CONV/MLP CONV/MLP CONV/MLP
```

## Starter v0.1
- Servidor central Java acepta clientes concurrentes con `ExecutorService` y distribuye solicitudes **completas** en round-robin a workers registrados en argumentos.
- Worker Java y Python aceptan múltiples conexiones y evalúan `PING`, `CONV` y `MLP`.
- Worker C# ofrece el mismo protocolo (debe compilarse y validarse en una máquina con .NET 8).
- Cliente CLI Java para pruebas de comunicación.

## Pendiente para satisfacer íntegramente el parcial
- Implementar CNN funcional entrenada o con pesos justificados + módulo imagen + interfaz Swing.
- Partir un único cálculo de convolución o MLP en micro chunks y juntar resultados, no solo balancear solicitudes completas.
- Permitir modos secuencial, paralelo, distribuido con un core por nodo y paralelo-distribuido con múltiples cores por nodo.
- Desplegar en máquinas conectadas en red y comparar métricas con cargas equivalentes.
- Producción de informe/presentación PDF y ZIP depurado.

## Nota
No usar este round-robin básico como evidencia suficiente de *paralelismo distribuido dentro de la CNN*. Es únicamente la primera integración de red.
