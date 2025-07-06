import hashlib

# Diccionario simulado de usuarios y contraseñas hasheadas
usuarios = {
    "admin": hashlib.sha256("1234".encode()).hexdigest(),
    "user": hashlib.sha256("abcd".encode()).hexdigest()
}

# Validar usuario y contraseña
def login(usuario, contraseña):
    contraseña_hash = hashlib.sha256(contraseña.encode()).hexdigest()
    if usuario in usuarios and usuarios[usuario] == contraseña_hash:
        return True
    return False

# Crear token simulado
def generar_token(usuario):
    import base64
    from datetime import datetime
    raw_token = f"{usuario}:{datetime.now()}"
    return base64.b64encode(raw_token.encode()).decode()
