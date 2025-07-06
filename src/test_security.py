from security.encryption import guardar_clave, cargar_clave, cifrar, descifrar
from security.hash_utils import crear_hash_sha256, crear_hash_md5
from security.auth import login, generar_token

# ======= Prueba de cifrado =========

guardar_clave()
clave = cargar_clave()

mensaje = "Hola mundo"
cifrado = cifrar(mensaje, clave)
descifrado = descifrar(cifrado, clave)

print("🔐 Cifrado:", cifrado)
print("🔓 Descifrado:", descifrado)

# ======= Prueba de hashes =========
print("📦 SHA256:", crear_hash_sha256(mensaje))
print("📦 MD5:", crear_hash_md5(mensaje))

# ======= Prueba de autenticación =========
print("✅ Login correcto (admin/1234):", login("admin", "1234"))
print("❌ Login incorrecto (user/malacontraseña):", login("user", "malacontraseña"))
print("🎫 Token generado:", generar_token("admin"))