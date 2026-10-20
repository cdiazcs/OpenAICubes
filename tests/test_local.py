import socket
import struct
import random

def send_msg(sock, msg):
    # Empaquetar longitud (4 bytes big-endian) + mensaje
    sock.sendall(struct.pack('!I', len(msg.encode('utf-8'))) + msg.encode('utf-8'))
    # Leer respuesta
    length_bytes = sock.recv(4)
    if not length_bytes: return None
    length = struct.unpack('!I', length_bytes)[0]
    return sock.recv(length).decode('utf-8')

def test_identify():
    # Creamos una imagen falsa de 16x16 (256 valores aleatorios entre 0 y 1)
    # Simula una imagen en escala de grises normalizada
    imagen_falsa = [round(random.uniform(0.0, 1.0), 2) for _ in range(256)]
    imagen_csv = ",".join(str(x) for x in imagen_falsa)
    
    # Armamos el mensaje del protocolo
    mensaje = f"IDENTIFY|req_123|{imagen_csv}"
    
    print("1. Conectando al worker en el puerto 9102...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1', 9102))
            print("2. Conexión establecida. Enviando imagen 16x16 (256 píxeles)...")
            
            respuesta = send_msg(s, mensaje)
            
            print("\n--- RESULTADO ---")
            print("Respuesta del servidor:", respuesta)
            
            # Validar la respuesta
            if respuesta and respuesta.startswith("RESULT|"):
                partes = respuesta.split('|')
                probs = partes[4].split(',')
                print(f"Clase 1 (Circulo): {probs[0]}")
                print(f"Clase 2 (Cruz): {probs[1]}")
                print(f"Clase 3 (Triangulo): {probs[2]}")
                print(f"Suma de probabilidades: {sum(float(p) for p in probs)}")
                print("\n¡ÉXITO! El pipeline completo (Conv -> Pool -> MLP) funcionó.")
            else:
                print("\nHubo un error en el worker. Revisa la consola donde lo ejecutaste.")
                
    except ConnectionRefusedError:
        print("ERROR: El worker no está encendido. Ejecuta primero 'python workers/python/worker.py'")

if __name__ == "__main__":
    test_identify()