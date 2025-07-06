import hashlib

def crear_hash_sha256(texto):
    sha_signature = hashlib.sha256(texto.encode()).hexdigest()
    return sha_signature

def crear_hash_md5(texto):
    md5_signature = hashlib.md5(texto.encode()).hexdigest()
    return md5_signature
