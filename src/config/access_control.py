# src/config/access_control.py
from enum import Enum, auto
import streamlit as st # Importa Streamlit para usar st.session_state, st.success, etc.
from functools import wraps # Todavía útil para envolver funciones si es necesario

# 1. Definición de Roles
class UserRole(Enum):
    ADMIN = "admin"
    REGULAR_USER = "regular_user"
    # Puedes añadir otros roles si es necesario, ej. PREMIUM_USER = "premium"

# 2. Definición de Permisos
class Permission(Enum):
    SCAN_FILES = auto()         # Permiso para iniciar un escaneo
    DELETE_FILES = auto()       # Permiso para eliminar archivos encontrados
    VIEW_REPORTS = auto()       # Permiso para ver reportes de limpieza
    MANAGE_USERS = auto()       # Permiso para administrar otros usuarios
    MANAGE_SETTINGS = auto()    # Permiso para cambiar la configuración global del limpiador
    UPLOAD_FILES = auto()       # Si el limpiador permite subir archivos para escanear
    DOWNLOAD_LOGS = auto()      # Descargar logs de actividad del sistema

# 3. Mapeo de Roles a Permisos
# Este diccionario define qué permisos tiene cada rol.
ROLE_PERMISSIONS = {
    UserRole.ADMIN: {
        Permission.SCAN_FILES,
        Permission.DELETE_FILES,
        Permission.VIEW_REPORTS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_SETTINGS,
        Permission.UPLOAD_FILES,
        Permission.DOWNLOAD_LOGS
    },
    UserRole.REGULAR_USER: {
        Permission.SCAN_FILES,
        Permission.DELETE_FILES,    # Asumiendo que pueden eliminar sus propios archivos
        Permission.VIEW_REPORTS,    # Solo sus propios reportes
        Permission.UPLOAD_FILES
    }
}

# --- Funciones de Utilidad para Autenticación/Autorización en Streamlit ---

def initialize_session_state():
    """
    Inicializa el estado de la sesión de Streamlit para el control de acceso.
    Debe llamarse al inicio de tu script Streamlit (dataloop_ui.py).
    """
    # Asegura que estas claves existan en st.session_state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_data" not in st.session_state:
        st.session_state.user_data = None # {'id': ..., 'username': ..., 'role': ...}
    # Puedes añadir otras variables de estado aquí, ej. para resultados de escaneo
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = []
    if "scanned_files_list" not in st.session_state:
        st.session_state.scanned_files_list = []
    if "duplicates" not in st.session_state: # Para almacenar los grupos de duplicados
        st.session_state.duplicates = []
    if "total_files" not in st.session_state: # Para resumen del escaneo
        st.session_state.total_files = 0
    if "total_size" not in st.session_state:
        st.session_state.total_size = 0
    if "total_duplicate_groups" not in st.session_state:
        st.session_state.total_duplicate_groups = 0
    if "duplicates_size" not in st.session_state:
        st.session_state.duplicates_size = 0


def get_current_user_data():
    """
    Obtiene los datos del usuario actual del st.session_state de Streamlit.
    Convierte el 'role' guardado de string a UserRole Enum.
    """
    user_data = st.session_state.get('user_data', None)

    if user_data and 'role' in user_data and isinstance(user_data['role'], str):
        try:
            # Intenta convertir el string del rol de nuevo a un objeto UserRole Enum
            user_data['role'] = UserRole(user_data['role'])
        except ValueError:
            # Si el string del rol no es válido, puedes manejar el error
            # Por ejemplo, asignar un rol por defecto o None
            user_data['role'] = None 
            st.error("Error: Rol de usuario inválido en la sesión.") # Esto aparecería en Streamlit

    return user_data

def get_user_roles(user_data):
    """
    Devuelve los roles de un usuario dado a partir de sus datos.
    """
    if not user_data or not user_data.get("role"):
        return set() # Usuario no logueado o sin rol

    # Si el rol ya es un Enum, lo usamos directamente. Si es un string, intentamos convertirlo.
    if isinstance(user_data["role"], UserRole):
        return {user_data["role"]}
    elif isinstance(user_data["role"], str):
        try:
            return {UserRole(user_data["role"])}
        except ValueError:
            return set() # Rol no reconocido
    return set()

def has_permission(user_data, required_permission: Permission) -> bool:
    """
    Verifica si un usuario (representado por user_data) tiene un permiso específico.
    """
    if not user_data:
        return False # Si no hay datos de usuario, no tiene permisos

    user_roles = get_user_roles(user_data)
    for role in user_roles:
        if required_permission in ROLE_PERMISSIONS.get(role, set()):
            return True
    return False

# --- Funciones para Autenticación (Login/Logout) en Streamlit ---

def login_user(username, role: UserRole):
    """
    Simula el login de un usuario, guardando los datos en st.session_state.
    """
    st.session_state.logged_in = True
    st.session_state.user_data = {
        'id': 1 if role == UserRole.ADMIN else 2, # ID dummy, puedes usar un UUID real si tienes DB
        'username': username,
        'role': role.value # Guardar el valor string del Enum
    }
    st.success(f"Sesión iniciada como {username} ({role.value}).")

def logout_user():
    """
    Cierra la sesión del usuario, eliminando los datos de st.session_state.
    """
    st.session_state.logged_in = False
    st.session_state.user_data = None
    st.info("Sesión cerrada.")
