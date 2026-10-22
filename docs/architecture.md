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

