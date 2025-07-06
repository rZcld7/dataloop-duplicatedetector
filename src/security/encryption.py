from cryptography.fernet import Fernet

# Generar clave (esto se hace una vez y se guarda)
def generar_clave():
    return Fernet.generate_key()

# Guardar clave en archivo
def guardar_clave(nombre_archivo="clave.key"):
    clave = generar_clave()
    with open(nombre_archivo, "wb") as file:
        file.write(clave)

# Cargar clave
def cargar_clave(nombre_archivo="clave.key"):
    with open(nombre_archivo, "rb") as file:
        return file.read()

# Cifrar mensaje
def cifrar(mensaje, clave):
    f = Fernet(clave)
    return f.encrypt(mensaje.encode())

# Descifrar mensaje
def descifrar(token, clave):
    f = Fernet(clave)
    return f.decrypt(token).decode()
