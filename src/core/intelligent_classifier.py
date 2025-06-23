"""
Este módulo implementa un clasificador inteligente de archivos que analiza su contenido y metadatos 
para determinar su tipo, nivel de riesgo y acción recomendada (mantener, revisar o eliminar). 
También agrupa archivos similares mediante clustering y genera reportes detallados en formato JSON o CSV.
"""

import os
import time
import json
import hashlib
import platform
import mimetypes
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class FileClassification:
    """Resultado de clasificación de un archivo"""
    filepath: str
    file_type: str
    category: str
    confidence: float
    safety_score: float
    recommended_action: str
    cluster_id: int
    features: Dict[str, any]
    risk_factors: List[str]
    processing_time: float = 0.0

class RealTfidfVectorizer:
    """Implementación real de TF-IDF Vectorizer"""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_values = {}
        self.fitted = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokeniza texto en palabras"""
        # Convertir a minúsculas y dividir por espacios y caracteres especiales
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [word.strip() for word in text.split() if word.strip()]
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calcula Term Frequency"""
        tf_dict = {}
        total_tokens = len(tokens)
        
        for token in tokens:
            tf_dict[token] = tf_dict.get(token, 0) + 1
        
        # Normalizar por total de tokens
        for token in tf_dict:
            tf_dict[token] = tf_dict[token] / total_tokens
            
        return tf_dict
    
    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """Entrena el vectorizador y transforma documentos"""
        if not documents:
            return []
        
        # Tokenizar todos los documentos
        all_tokens = []
        doc_tokens = []
        
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_tokens.append(tokens)
            all_tokens.extend(tokens)
        
        # Crear vocabulario con los términos más frecuentes
        token_counts = Counter(all_tokens)
        most_common = token_counts.most_common(self.max_features)
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(most_common)}
        
        # Calcular IDF
        total_docs = len(documents)
        for token in self.vocabulary:
            # Contar en cuántos documentos aparece el token
            doc_freq = sum(1 for tokens in doc_tokens if token in tokens)
            self.idf_values[token] = math.log(total_docs / (doc_freq + 1))
        
        self.fitted = True
        
        # Transformar documentos
        return self.transform(documents)
    
    def transform(self, documents: List[str]) -> List[List[float]]:
        """Transforma documentos a vectores TF-IDF"""
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        vectors = []
        vocab_size = len(self.vocabulary)
        
        for doc in documents:
            tokens = self._tokenize(doc)
            tf_dict = self._compute_tf(tokens)
            
            # Crear vector TF-IDF
            vector = [0.0] * vocab_size
            
            for token, tf_value in tf_dict.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    idf_value = self.idf_values[token]
                    vector[idx] = tf_value * idf_value
            
            vectors.append(vector)
        
        return vectors

class RealKMeans:
    """Implementación real de K-Means clustering"""
    
    def __init__(self, n_clusters: int = 5, max_iters: int = 100, random_seed: int = 42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_seed = random_seed
        self.centroids = []
        self.labels = []
    
    def _euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calcula distancia euclidiana entre dos puntos"""
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
    
    def _initialize_centroids(self, data: List[List[float]]) -> List[List[float]]:
        """Inicializa centroides aleatoriamente"""
        import random
        random.seed(self.random_seed)
        
        if not data:
            return []
        
        n_features = len(data[0])
        centroids = []
        
        # K-means++ initialization para mejor convergencia
        centroids.append(random.choice(data))
        
        for _ in range(1, self.n_clusters):
            distances = []
            for point in data:
                min_dist = min(self._euclidean_distance(point, centroid) for centroid in centroids)
                distances.append(min_dist ** 2)
            
            # Seleccionar siguiente centroide con probabilidad proporcional a distancia
            total_dist = sum(distances)
            if total_dist > 0:
                probabilities = [d / total_dist for d in distances]
                cumulative = []
                cum_sum = 0
                for p in probabilities:
                    cum_sum += p
                    cumulative.append(cum_sum)
                
                rand_val = random.random()
                for i, cum_prob in enumerate(cumulative):
                    if rand_val <= cum_prob:
                        centroids.append(data[i])
                        break
            else:
                centroids.append(random.choice(data))
        
        return centroids
    
    def fit_predict(self, data: List[List[float]]) -> List[int]:
        """Entrena K-means y retorna etiquetas de cluster"""
        if not data or len(data) < self.n_clusters:
            return [0] * len(data)
        
        # Inicializar centroides
        self.centroids = self._initialize_centroids(data)
        
        for iteration in range(self.max_iters):
            # Asignar puntos a clusters
            new_labels = []
            for point in data:
                distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
                cluster_id = distances.index(min(distances))
                new_labels.append(cluster_id)
            
            # Verificar convergencia
            if iteration > 0 and new_labels == self.labels:
                break
            
            self.labels = new_labels
            
            # Actualizar centroides
            new_centroids = []
            for cluster_id in range(self.n_clusters):
                cluster_points = [data[i] for i, label in enumerate(self.labels) if label == cluster_id]
                
                if cluster_points:
                    # Calcular centroide promedio
                    n_features = len(cluster_points[0])
                    new_centroid = []
                    for feature_idx in range(n_features):
                        avg_value = sum(point[feature_idx] for point in cluster_points) / len(cluster_points)
                        new_centroid.append(avg_value)
                    new_centroids.append(new_centroid)
                else:
                    # Mantener centroide anterior si no hay puntos asignados
                    new_centroids.append(self.centroids[cluster_id])
            
            self.centroids = new_centroids
        
        return self.labels

class IntelligentClassifier:
    """Clasificador inteligente de archivos usando ML real"""
    
    def __init__(self, max_workers: int = 4):
        self.vectorizer = RealTfidfVectorizer(max_features=500)
        self.kmeans = RealKMeans(n_clusters=6)
        self.max_workers = min(max_workers, os.cpu_count() or 4)
        
        # Detectar sistema operativo
        self.system = platform.system().lower()
        
        # Configurar mimetypes
        mimetypes.init()
        
        # Categorías de archivos extendidas
        self.file_categories = {
            'images': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.ico', '.svg'},
            'videos': {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp'},
            'documents': {'.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx'},
            'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'},
            'archives': {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'},
            'executables': {'.exe', '.msi', '.dmg', '.deb', '.rpm', '.appimage', '.app'},
            'code': {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.json', '.xml'},
            'fonts': {'.ttf', '.otf', '.woff', '.woff2'},
            'others': set()
        }
        
        # Patrones de riesgo por sistema operativo
        self.system_patterns = self._get_system_patterns()
        
        # Cache para mejorar rendimiento
        self._feature_cache = {}
        self._cache_lock = threading.Lock()
        
        self.is_trained = False
        self.training_stats = {}
        
        print(f"🤖 IntelligentClassifier inicializado para {self.system}")
        print(f"   Workers: {self.max_workers}")
        print(f"   Categorías: {len(self.file_categories)}")
    
    def _get_system_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Obtiene patrones específicos del sistema operativo"""
        patterns = {
            'windows': {
                'system_dirs': ['windows', 'system32', 'program files', 'programdata', 'appdata'],
                'temp_dirs': ['temp', 'tmp', '%temp%', 'temporary internet files'],
                'risky_locations': ['desktop', 'downloads', 'documents\\downloads'],
                'safe_extensions': {'.txt', '.pdf', '.jpg', '.png'},
                'risky_extensions': {'.exe', '.bat', '.cmd', '.com', '.scr', '.vbs'}
            },
            'linux': {
                'system_dirs': ['usr', 'etc', 'var', 'opt', 'sys', 'proc'],
                'temp_dirs': ['tmp', '/tmp', '/var/tmp'],
                'risky_locations': ['downloads', 'desktop'],
                'safe_extensions': {'.txt', '.pdf', '.jpg', '.png'},
                'risky_extensions': {'.sh', '.run', '.deb', '.rpm'}
            },
            'darwin': {  # macOS
                'system_dirs': ['system', 'library', 'applications'],
                'temp_dirs': ['tmp', '/tmp', '/var/tmp'],
                'risky_locations': ['downloads', 'desktop'],
                'safe_extensions': {'.txt', '.pdf', '.jpg', '.png'},
                'risky_extensions': {'.dmg', '.pkg', '.app'}
            }
        }
        
        return patterns.get(self.system, patterns['linux'])
    
    def extract_features(self, filepath: str, use_cache: bool = True) -> Dict[str, any]:
        """Extrae características avanzadas de un archivo de forma eficiente"""
        
        # Verificar cache
        if use_cache:
            with self._cache_lock:
                if filepath in self._feature_cache:
                    cached_features, cache_time = self._feature_cache[filepath]
                    if time.time() - cache_time < 300:  # Cache válido por 5 minutos
                        return cached_features
        
        path = Path(filepath)
        
        if not path.exists():
            return {}
        
        try:
            start_time = time.time()
            stat_info = path.stat()
            
            # Características básicas
            features = {
                'filename': path.name,
                'extension': path.suffix.lower(),
                'size_bytes': stat_info.st_size,
                'size_mb': stat_info.st_size / (1024 * 1024),
                'size_category': self._get_size_category(stat_info.st_size),
            }
            
            # Características de tiempo
            current_time = time.time()
            features.update({
                'creation_time': getattr(stat_info, 'st_birthtime', stat_info.st_ctime),
                'modified_time': stat_info.st_mtime,
                'access_time': stat_info.st_atime,
                'age_days': (current_time - stat_info.st_mtime) / (24 * 3600),
                'age_category': self._get_age_category((current_time - stat_info.st_mtime) / (24 * 3600)),
            })
            
            # Características de ubicación
            features.update({
                'parent_dir': path.parent.name.lower(),
                'depth_level': len(path.parts),
                'full_path_lower': str(path).lower(),
                'in_system_dir': self._is_system_directory(str(path.parent)),
                'in_temp_dir': self._is_temp_directory(str(path.parent)),
                'in_user_dir': self._is_user_directory(str(path)),
            })
            
            # Características de nombre
            stem = path.stem
            features.update({
                'has_numbers': any(c.isdigit() for c in stem),
                'has_copy_pattern': self._has_copy_pattern(stem),
                'name_length': len(stem),
                'special_chars_count': sum(1 for c in stem if not c.isalnum() and c not in ' -_'),
                'name_complexity': self._calculate_name_complexity(stem),
            })
            
            # Características de tipo
            extension = path.suffix.lower()
            features.update({
                'file_category': self._get_file_category(extension),
                'mime_type': self._get_mime_type(filepath),
                'is_binary': self._is_binary_file(filepath),
                'is_text': self._is_text_file(extension),
            })
            
            # Características de seguridad
            features.update(self._extract_security_features(path))
            
            # Características de acceso
            features.update(self._extract_access_features(path, stat_info))
            
            features['extraction_time'] = time.time() - start_time
            
            # Guardar en cache
            if use_cache:
                with self._cache_lock:
                    self._feature_cache[filepath] = (features, time.time())
                    # Limpiar cache si es muy grande
                    if len(self._feature_cache) > 1000:
                        oldest_key = min(self._feature_cache.keys(), 
                                       key=lambda k: self._feature_cache[k][1])
                        del self._feature_cache[oldest_key]
            
            return features
            
        except Exception as e:
            print(f"Error extrayendo características de {filepath}: {e}")
            return {'error': str(e), 'filepath': filepath}
    
    def _get_size_category(self, size_bytes: int) -> str:
        """Categoriza archivos por tamaño"""
        if size_bytes == 0:
            return 'empty'
        elif size_bytes < 1024:  # < 1KB
            return 'tiny'
        elif size_bytes < 1024 * 1024:  # < 1MB
            return 'small'
        elif size_bytes < 100 * 1024 * 1024:  # < 100MB
            return 'medium'
        elif size_bytes < 1024 * 1024 * 1024:  # < 1GB
            return 'large'
        else:
            return 'huge'
    
    def _get_age_category(self, age_days: float) -> str:
        """Categoriza archivos por edad"""
        if age_days < 1:
            return 'today'
        elif age_days < 7:
            return 'this_week'
        elif age_days < 30:
            return 'this_month'
        elif age_days < 90:
            return 'this_quarter'
        elif age_days < 365:
            return 'this_year'
        else:
            return 'old'
    
    def _get_file_category(self, extension: str) -> str:
        """Determina la categoría del archivo por extensión"""
        for category, extensions in self.file_categories.items():
            if extension in extensions:
                return category
        return 'others'
    
    def _get_mime_type(self, filepath: str) -> str:
        """Obtiene el tipo MIME del archivo"""
        try:
            mime_type, _ = mimetypes.guess_type(filepath)
            return mime_type or 'unknown'
        except:
            return 'unknown'
    
    def _is_binary_file(self, filepath: str, sample_size: int = 1024) -> bool:
        """Verifica si un archivo es binario"""
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(sample_size)
                if not chunk:
                    return False
                
                # Buscar bytes nulos
                return b'\x00' in chunk
        except:
            return True  # Asumir binario si no se puede leer
    
    def _is_text_file(self, extension: str) -> bool:
        """Verifica si un archivo es de texto"""
        text_extensions = {'.txt', '.md', '.rst', '.log', '.cfg', '.ini', '.json', '.xml', '.csv'}
        return extension in text_extensions
    
    def _is_system_directory(self, directory: str) -> bool:
        """Verifica si es un directorio del sistema"""
        directory_lower = directory.lower()
        system_dirs = self.system_patterns['system_dirs']
        return any(sys_dir in directory_lower for sys_dir in system_dirs)
    
    def _is_temp_directory(self, directory: str) -> bool:
        """Verifica si es un directorio temporal"""
        directory_lower = directory.lower()
        temp_dirs = self.system_patterns['temp_dirs']
        return any(temp_dir in directory_lower for temp_dir in temp_dirs)
    
    def _is_user_directory(self, filepath: str) -> bool:
        """Verifica si está en directorio de usuario"""
        try:
            home_dir = str(Path.home()).lower()
            return filepath.lower().startswith(home_dir)
        except:
            return False
    
    def _has_copy_pattern(self, filename: str) -> bool:
        """Verifica patrones de archivo copiado"""
        patterns = ['copy', 'duplicate', '(1)', '(2)', '(3)', '_copy', '-copy', 'backup']
        filename_lower = filename.lower()
        return any(pattern in filename_lower for pattern in patterns)
    
    def _calculate_name_complexity(self, filename: str) -> float:
        """Calcula complejidad del nombre de archivo"""
        if not filename:
            return 0.0
        
        # Factores que aumentan complejidad
        factors = {
            'length': min(len(filename) / 50, 1.0),  # Nombres muy largos
            'special_chars': min(sum(1 for c in filename if not c.isalnum()) / len(filename), 0.5),
            'numbers': min(sum(1 for c in filename if c.isdigit()) / len(filename), 0.3),
            'mixed_case': 0.1 if any(c.isupper() for c in filename) and any(c.islower() for c in filename) else 0,
        }
        
        return sum(factors.values()) / len(factors)
    
    def _extract_security_features(self, path: Path) -> Dict[str, any]:
        """Extrae características relacionadas con seguridad"""
        try:
            filepath_str = str(path)
            filename = path.name
            extension = path.suffix.lower()
            
            features = {
                'is_hidden': filename.startswith('.') or (self.system == 'windows' and self._is_windows_hidden(filepath_str)),
                'is_executable': extension in self.system_patterns['risky_extensions'],
                'is_safe_extension': extension in self.system_patterns['safe_extensions'],
                'suspicious_name': self._has_suspicious_name(filename),
                'in_risky_location': self._is_risky_location(str(path.parent)),
                'has_double_extension': self._has_double_extension(filename),
            }
            
            # Verificar permisos (Unix-like systems)
            if self.system in ['linux', 'darwin']:
                try:
                    stat_info = path.stat()
                    features['is_world_writable'] = bool(stat_info.st_mode & 0o002)
                    features['is_setuid'] = bool(stat_info.st_mode & 0o4000)
                    features['is_setgid'] = bool(stat_info.st_mode & 0o2000)
                except:
                    features.update({'is_world_writable': False, 'is_setuid': False, 'is_setgid': False})
            
            return features
            
        except Exception as e:
            return {'security_error': str(e)}
    
    def _is_windows_hidden(self, filepath: str) -> bool:
        """Verifica si un archivo está oculto en Windows"""
        if self.system != 'windows':
            return False
        
        try:
            import ctypes
            attrs = ctypes.windll.kernel32.GetFileAttributesW(filepath)
            return attrs != -1 and bool(attrs & 2)  # FILE_ATTRIBUTE_HIDDEN
        except:
            return False
    
    def _has_suspicious_name(self, filename: str) -> bool:
        """Verifica si el nombre del archivo es sospechoso"""
        suspicious_patterns = [
            'virus', 'malware', 'trojan', 'worm', 'rootkit',
            'hack', 'crack', 'keygen', 'patch', 'loader',
            'bitcoin', 'crypto', 'miner', 'ransomware'
        ]
        filename_lower = filename.lower()
        return any(pattern in filename_lower for pattern in suspicious_patterns)
    
    def _is_risky_location(self, directory: str) -> bool:
        """Verifica si la ubicación es de riesgo"""
        directory_lower = directory.lower()
        risky_locations = self.system_patterns['risky_locations']
        return any(loc in directory_lower for loc in risky_locations)
    
    def _has_double_extension(self, filename: str) -> bool:
        """Verifica extensiones dobles sospechosas"""
        parts = filename.split('.')
        if len(parts) < 3:
            return False
        
        # Buscar patrones como file.pdf.exe
        risky_extensions = self.system_patterns['risky_extensions']
        return any(f'.{part}' in risky_extensions for part in parts[-2:])
    
    def _extract_access_features(self, path: Path, stat_info) -> Dict[str, any]:
        """Extrae características de acceso al archivo"""
        try:
            current_time = time.time()
            
            features = {
                'days_since_access': (current_time - stat_info.st_atime) / (24 * 3600),
                'days_since_modified': (current_time - stat_info.st_mtime) / (24 * 3600),
                'access_frequency_score': self._calculate_access_frequency(stat_info),
            }
            
            # Verificar si el archivo puede ser leído/escrito
            features.update({
                'readable': os.access(path, os.R_OK),
                'writable': os.access(path, os.W_OK),
                'executable': os.access(path, os.X_OK),
            })
            
            return features
            
        except Exception as e:
            return {'access_error': str(e)}
    
    def _calculate_access_frequency(self, stat_info) -> float:
        """Calcula un score de frecuencia de acceso"""
        current_time = time.time()
        
        # Diferencia entre modificación y acceso
        mod_access_diff = abs(stat_info.st_mtime - stat_info.st_atime)
        
        # Si son muy similares, probablemente se accede poco
        if mod_access_diff < 3600:  # 1 hora
            return 0.1
        elif mod_access_diff < 86400:  # 1 día
            return 0.3
        elif mod_access_diff < 604800:  # 1 semana
            return 0.6
        else:
            return 1.0
    
    def calculate_safety_score(self, features: Dict[str, any]) -> Tuple[float, List[str]]:
        """Calcula un score de seguridad mejorado para eliminar el archivo"""
        
        if 'error' in features:
            return 0.0, [f"Error: {features['error']}"]
        
        score = 0.5  # Neutro por defecto
        risk_factors = []
        
        try:
            # Factor 1: Edad del archivo (25% del peso)
            age_days = features.get('age_days', 0)
            age_weight = 0.25
            
            if age_days > 730:  # Más de 2 años
                score += age_weight * 0.8
            elif age_days > 365:  # Más de 1 año
                score += age_weight * 0.6
            elif age_days > 90:  # Más de 3 meses
                score += age_weight * 0.3
            elif age_days < 1:  # Menos de 1 día
                score -= age_weight * 0.8
                risk_factors.append("Archivo muy reciente")
            elif age_days < 7:  # Menos de una semana
                score -= age_weight * 0.4
                risk_factors.append("Archivo reciente")
            
            # Factor 2: Ubicación del archivo (20% del peso)
            location_weight = 0.20
            
            if features.get('in_temp_dir', False):
                score += location_weight * 1.0
            elif features.get('in_system_dir', False):
                score -= location_weight * 1.0
                risk_factors.append("Archivo en directorio del sistema")
            elif features.get('in_risky_location', False):
                score += location_weight * 0.3
            
            # Factor 3: Patrones en el nombre (15% del peso)
            name_weight = 0.15
            
            if features.get('has_copy_pattern', False):
                score += name_weight * 0.8
            
            name_complexity = features.get('name_complexity', 0.5)
            if name_complexity > 0.7:
                score += name_weight * 0.4
            
            # Factor 4: Tipo y categoría de archivo (20% del peso)
            type_weight = 0.20
            
            file_category = features.get('file_category', 'others')
            extension = features.get('extension', '')
            
            if features.get('is_executable', False):
                score -= type_weight * 0.8
                risk_factors.append("Archivo ejecutable")
            elif features.get('is_safe_extension', False):
                score += type_weight * 0.4
            elif file_category in ['images', 'audio']:
                score += type_weight * 0.2
            elif file_category == 'others':
                score += type_weight * 0.1
            
            # Factor 5: Tamaño del archivo (10% del peso)
            size_weight = 0.10
            size_category = features.get('size_category', 'medium')
            
            if size_category == 'empty':
                score += size_weight * 1.0
            elif size_category == 'tiny':
                score += size_weight * 0.6
            elif size_category == 'huge':
                score -= size_weight * 0.4
                risk_factors.append("Archivo muy grande")
            
            # Factor 6: Acceso y uso (10% del peso)
            access_weight = 0.10
            
            days_since_access = features.get('days_since_access', 0)
            if days_since_access > 365:  # No accedido en más de 1 año
                score += access_weight * 0.8
            elif days_since_access > 90:  # No accedido en más de 3 meses
                score += access_weight * 0.4
            elif days_since_access < 1:  # Accedido recientemente
                score -= access_weight * 0.6
                risk_factors.append("Archivo accedido recientemente")
            
            # Factores de seguridad adicionales
            if features.get('suspicious_name', False):
                score -= 0.3
                risk_factors.append("Nombre sospechoso")
            
            if features.get('has_double_extension', False):
                score -= 0.2
                risk_factors.append("Extensión doble sospechosa")
            
            if features.get('is_hidden', False) and not features.get('in_system_dir', False):
                score += 0.1  # Archivos ocultos no del sistema pueden ser temporales
            
            # Normalizar score entre 0 y 1
            score = max(0.0, min(1.0, score))
            
            return score, risk_factors
            
        except Exception as e:
            return 0.5, [f"Error calculando safety score: {str(e)}"]
        

    def get_recommendation(self, safety_score: float, features: Dict[str, any], risk_factors: List[str]) -> str:
        """Genera recomendación basada en el safety score y características"""
        
        try:
            # Verificar factores críticos
            critical_factors = [
                "Archivo en directorio del sistema",
                "Archivo accedido recientemente",
                "Archivo muy reciente"
            ]
            
            has_critical_risk = any(factor in risk_factors for factor in critical_factors)
            
            if has_critical_risk:
                return "KEEP"
            
            # Decisiones basadas en safety score
            if safety_score >= 0.8:
                return "DELETE"
            elif safety_score >= 0.6:
                # Verificar factores adicionales
                file_category = features.get('file_category', 'others')
                age_days = features.get('age_days', 0)
                size_mb = features.get('size_mb', 0)
                
                if file_category in ['images', 'videos'] and size_mb > 100:
                    return "ARCHIVE"
                elif age_days > 180 and file_category != 'executables':
                    return "DELETE"
                else:
                    return "ARCHIVE"
            elif safety_score >= 0.4:
                return "REVIEW"
            else:
                return "KEEP"
                
        except Exception as e:
            print(f"Error generando recomendación: {e}")
            return "REVIEW"
    
    def classify_file(self, filepath: str) -> FileClassification:
        """Clasifica un archivo individual"""
        start_time = time.time()
        
        try:
            # Extraer características
            features = self.extract_features(filepath)
            
            if 'error' in features:
                return FileClassification(
                    filepath=filepath,
                    file_type='unknown',
                    category='error',
                    confidence=0.0,
                    safety_score=0.0,
                    recommended_action='REVIEW',
                    cluster_id=-1,
                    features=features,
                    risk_factors=[features['error']],
                    processing_time=time.time() - start_time
                )
            
            # Calcular safety score
            safety_score, risk_factors = self.calculate_safety_score(features)
            
            # Generar recomendación
            recommendation = self.get_recommendation(safety_score, features, risk_factors)
            
            # Determinar tipo de archivo y categoría
            file_type = features.get('mime_type', 'unknown')
            category = features.get('file_category', 'others')
            
            # Calcular confianza basada en características disponibles
            confidence = self._calculate_confidence(features, risk_factors)
            
            return FileClassification(
                filepath=filepath,
                file_type=file_type,
                category=category,
                confidence=confidence,
                safety_score=safety_score,
                recommended_action=recommendation,
                cluster_id=0,  # Se asignará durante el clustering
                features=features,
                risk_factors=risk_factors,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return FileClassification(
                filepath=filepath,
                file_type='error',
                category='error',
                confidence=0.0,
                safety_score=0.0,
                recommended_action='REVIEW',
                cluster_id=-1,
                features={'error': str(e)},
                risk_factors=[f"Error de clasificación: {str(e)}"],
                processing_time=time.time() - start_time
            )
    
    def _calculate_confidence(self, features: Dict[str, any], risk_factors: List[str]) -> float:
        """Calcula el nivel de confianza en la clasificación"""
        base_confidence = 0.7
        
        # Reducir confianza por errores o datos faltantes
        if 'error' in features:
            return 0.1
        
        # Aumentar confianza con más características disponibles
        available_features = sum(1 for key, value in features.items() 
                               if value is not None and value != 'unknown')
        
        feature_bonus = min(available_features / 20, 0.2)
        
        # Reducir confianza por factores de riesgo críticos
        critical_risks = sum(1 for risk in risk_factors 
                           if any(critical in risk.lower() 
                                for critical in ['sistema', 'reciente', 'sospechoso']))
        
        risk_penalty = min(critical_risks * 0.1, 0.3)
        
        final_confidence = base_confidence + feature_bonus - risk_penalty
        return max(0.1, min(1.0, final_confidence))
    
    def classify_directory(self, directory_path: str, recursive: bool = True, 
                          file_patterns: Optional[List[str]] = None) -> List[FileClassification]:
        """Clasifica todos los archivos en un directorio"""
        
        print(f"🔍 Clasificando directorio: {directory_path}")
        print(f"   Recursivo: {recursive}")
        print(f"   Workers: {self.max_workers}")
        
        if file_patterns:
            print(f"   Patrones: {file_patterns}")
        
        start_time = time.time()
        
        # Encontrar todos los archivos
        file_paths = self._find_files(directory_path, recursive, file_patterns)
        total_files = len(file_paths)
        
        if total_files == 0:
            print("   ⚠ No se encontraron archivos")
            return []
        
        print(f"   📁 Archivos encontrados: {total_files}")
        
        # Clasificar archivos en paralelo
        classifications = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Enviar tareas
            future_to_path = {
                executor.submit(self.classify_file, filepath): filepath 
                for filepath in file_paths
            }
            
            # Procesar resultados
            processed = 0
            for future in as_completed(future_to_path):
                try:
                    classification = future.result()
                    classifications.append(classification)
                    
                    processed += 1
                    if processed % 100 == 0 or processed == total_files:
                        progress = (processed / total_files) * 100
                        print(f"   📊 Progreso: {processed}/{total_files} ({progress:.1f}%)")
                        
                except Exception as e:
                    filepath = future_to_path[future]
                    print(f"   ❌ Error procesando {filepath}: {e}")
                    classifications.append(FileClassification(
                        filepath=filepath,
                        file_type='error',
                        category='error',
                        confidence=0.0,
                        safety_score=0.0,
                        recommended_action='REVIEW',
                        cluster_id=-1,
                        features={'error': str(e)},
                        risk_factors=[f"Error de procesamiento: {str(e)}"],
                        processing_time=0.0
                    ))
        
        # Realizar clustering si hay suficientes archivos
        if len(classifications) > 5:
            classifications = self._perform_clustering(classifications)
        
        total_time = time.time() - start_time
        print(f"   ✅ Clasificación completada en {total_time:.2f}s")
        print(f"   📈 Promedio: {total_time/total_files:.3f}s por archivo")
        
        return classifications
    
    def _find_files(self, directory_path: str, recursive: bool = True, 
                   file_patterns: Optional[List[str]] = None) -> List[str]:
        """Encuentra archivos en un directorio"""
        
        file_paths = []
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"   ❌ Directorio no existe: {directory_path}")
            return []
        
        try:
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for path in directory.glob(pattern):
                if path.is_file():
                    # Aplicar filtros de patrones si se especifican
                    if file_patterns:
                        filename = path.name.lower()
                        if any(pattern.lower() in filename for pattern in file_patterns):
                            file_paths.append(str(path))
                    else:
                        file_paths.append(str(path))
                        
        except PermissionError as e:
            print(f"   ⚠ Sin permisos para acceder a {directory_path}: {e}")
        except Exception as e:
            print(f"   ❌ Error explorando {directory_path}: {e}")
        
        return file_paths
    
    def _perform_clustering(self, classifications: List[FileClassification]) -> List[FileClassification]:
        """Realiza clustering de archivos similares"""
        
        print("   🧠 Realizando clustering...")
        
        try:
            # Preparar datos para clustering
            documents = []
            valid_classifications = []
            
            for classification in classifications:
                if classification.category != 'error':
                    # Crear documento de características
                    features = classification.features
                    doc_parts = [
                        features.get('file_category', ''),
                        features.get('size_category', ''),
                        features.get('age_category', ''),
                        features.get('parent_dir', ''),
                        str(features.get('extension', '')),
                    ]
                    
                    doc = ' '.join(str(part) for part in doc_parts if part)
                    documents.append(doc)
                    valid_classifications.append(classification)
            
            if len(documents) < 2:
                return classifications
            
            # Vectorizar documentos
            vectors = self.vectorizer.fit_transform(documents)
            
            # Ajustar número de clusters
            n_clusters = min(6, max(2, len(documents) // 10))
            self.kmeans.n_clusters = n_clusters
            
            # Realizar clustering
            cluster_labels = self.kmeans.fit_predict(vectors)
            
            # Asignar clusters a clasificaciones
            for i, classification in enumerate(valid_classifications):
                classification.cluster_id = cluster_labels[i]
            
            # Marcar errores con cluster -1
            for classification in classifications:
                if classification.category == 'error':
                    classification.cluster_id = -1
            
            self.is_trained = True
            self.training_stats = {
                'total_files': len(classifications),
                'clustered_files': len(valid_classifications),
                'n_clusters': n_clusters,
                'training_time': time.time()
            }
            
            print(f"      ✅ Clustering completado: {n_clusters} clusters")
            
        except Exception as e:
            print(f"      ❌ Error en clustering: {e}")
            # Asignar cluster 0 por defecto
            for classification in classifications:
                if classification.cluster_id == 0:  # Solo si no se asignó antes
                    classification.cluster_id = 0
        
        return classifications
    
    def generate_report(self, classifications: List[FileClassification]) -> Dict[str, any]:
        """Genera un reporte detallado de la clasificación"""
        
        if not classifications:
            return {'error': 'No hay clasificaciones para reportar'}
        
        print("📊 Generando reporte...")
        
        # Estadísticas básicas
        total_files = len(classifications)
        total_size_mb = sum(c.features.get('size_mb', 0) for c in classifications 
                           if 'error' not in c.features)
        
        # Contadores por categoría
        categories = Counter(c.category for c in classifications)
        recommendations = Counter(c.recommended_action for c in classifications)
        file_types = Counter(c.file_type for c in classifications)
        
        # Estadísticas de safety scores
        safety_scores = [c.safety_score for c in classifications if c.safety_score > 0]
        avg_safety_score = sum(safety_scores) / len(safety_scores) if safety_scores else 0
        
        # Archivos por cluster
        clusters = Counter(c.cluster_id for c in classifications)
        
        # Top factores de riesgo
        all_risks = []
        for c in classifications:
            all_risks.extend(c.risk_factors)
        top_risks = Counter(all_risks).most_common(10)
        
        # Archivos recomendados para eliminación
        delete_candidates = [c for c in classifications if c.recommended_action == 'DELETE']
        delete_size_mb = sum(c.features.get('size_mb', 0) for c in delete_candidates)
        
        # Archivos sospechosos
        suspicious_files = [c for c in classifications 
                           if any('sospechoso' in risk.lower() for risk in c.risk_factors)]
        
        report = {
            'summary': {
                'total_files': total_files,
                'total_size_mb': round(total_size_mb, 2),
                'total_size_gb': round(total_size_mb / 1024, 2),
                'avg_safety_score': round(avg_safety_score, 3),
                'classification_time': sum(c.processing_time for c in classifications),
                'is_trained': self.is_trained
            },
            
            'categories': dict(categories),
            'recommendations': dict(recommendations),
            'file_types': dict(file_types.most_common(10)),
            'clusters': dict(clusters),
            
            'cleanup_potential': {
                'delete_candidates': len(delete_candidates),
                'delete_size_mb': round(delete_size_mb, 2),
                'delete_size_gb': round(delete_size_mb / 1024, 2),
                'space_savings_percent': round((delete_size_mb / total_size_mb) * 100, 1) if total_size_mb > 0 else 0
            },
            
            'risk_analysis': {
                'suspicious_files': len(suspicious_files),
                'top_risk_factors': [{'risk': risk, 'count': count} for risk, count in top_risks],
                'high_risk_files': len([c for c in classifications if c.safety_score < 0.3])
            },
            
            'performance': {
                'avg_processing_time': round(sum(c.processing_time for c in classifications) / total_files, 4),
                'training_stats': self.training_stats if hasattr(self, 'training_stats') else {}
            }
        }
        
        return report
    
    def save_results(self, classifications: List[FileClassification], 
                    output_path: str, format: str = 'json') -> bool:
        """Guarda los resultados en un archivo"""
        
        try:
            output_file = Path(output_path)
            
            if format.lower() == 'json':
                # Convertir a formato serializable
                data = {
                    'metadata': {
                        'timestamp': time.time(),
                        'system': self.system,
                        'total_files': len(classifications),
                        'classifier_version': '3.0'
                    },
                    'classifications': [asdict(c) for c in classifications],
                    'report': self.generate_report(classifications)
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            elif format.lower() == 'csv':
                # Crear CSV con información básica
                import csv
                
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = [
                        'filepath', 'file_type', 'category', 'confidence', 
                        'safety_score', 'recommended_action', 'cluster_id',
                        'size_mb', 'age_days', 'risk_factors'
                    ]
                    
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for c in classifications:
                        row = {
                            'filepath': c.filepath,
                            'file_type': c.file_type,
                            'category': c.category,
                            'confidence': round(c.confidence, 3),
                            'safety_score': round(c.safety_score, 3),
                            'recommended_action': c.recommended_action,
                            'cluster_id': c.cluster_id,
                            'size_mb': round(c.features.get('size_mb', 0), 2),
                            'age_days': round(c.features.get('age_days', 0), 1),
                            'risk_factors': '; '.join(c.risk_factors)
                        }
                        writer.writerow(row)
            
            else:
                raise ValueError(f"Formato no soportado: {format}")
            
            print(f"✅ Resultados guardados en: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
            return False
    
    def load_results(self, input_path: str) -> Optional[List[FileClassification]]:
        """Carga resultados desde un archivo JSON"""
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            classifications = []
            for item in data.get('classifications', []):
                # Reconstruir objeto FileClassification
                classification = FileClassification(**item)
                classifications.append(classification)
            
            print(f"✅ Cargados {len(classifications)} resultados desde {input_path}")
            return classifications
            
        except Exception as e:
            print(f"❌ Error cargando resultados: {e}")
            return None
    
    def get_cluster_summary(self, classifications: List[FileClassification]) -> Dict[int, Dict]:
        """Obtiene resumen de cada cluster"""
        
        cluster_data = defaultdict(list)
        
        # Agrupar por cluster
        for c in classifications:
            cluster_data[c.cluster_id].append(c)
        
        summaries = {}
        
        for cluster_id, cluster_files in cluster_data.items():
            if cluster_id == -1:  # Errores
                continue
                
            # Estadísticas del cluster
            categories = Counter(c.category for c in cluster_files)
            avg_safety = sum(c.safety_score for c in cluster_files) / len(cluster_files)
            total_size = sum(c.features.get('size_mb', 0) for c in cluster_files)
            
            # Archivos representativos
            representative_files = sorted(cluster_files, key=lambda x: x.confidence, reverse=True)[:5]
            
            summaries[cluster_id] = {
                'total_files': len(cluster_files),
                'main_categories': dict(categories.most_common(3)),
                'avg_safety_score': round(avg_safety, 3),
                'total_size_mb': round(total_size, 2),
                'representative_files': [
                    {
                        'filepath': Path(c.filepath).name,
                        'category': c.category,
                        'safety_score': round(c.safety_score, 3),
                        'recommendation': c.recommended_action
                    }
                    for c in representative_files
                ]
            }
        
        return summaries


def main():
    """Función principal de demostración"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clasificador Inteligente de Archivos')
    parser.add_argument('directory', help='Directorio a clasificar')
    parser.add_argument('--recursive', '-r', action='store_true', 
                       help='Búsqueda recursiva en subdirectorios')
    parser.add_argument('--output', '-o', default='classification_results.json',
                       help='Archivo de salida para resultados')
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='json',
                       help='Formato de salida')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Número de workers paralelos')
    parser.add_argument('--patterns', '-p', nargs='+',
                       help='Patrones de archivos a incluir')
    
    args = parser.parse_args()
    
    # Crear clasificador
    classifier = IntelligentClassifier(max_workers=args.workers)
    
    # Clasificar directorio
    classifications = classifier.classify_directory(
        args.directory, 
        recursive=args.recursive,
        file_patterns=args.patterns
    )
    
    if not classifications:
        print("❌ No se pudieron clasificar archivos")
        return
    
    # Generar y mostrar reporte
    report = classifier.generate_report(classifications)
    
    print("\n" + "="*60)
    print("📊 REPORTE DE CLASIFICACIÓN")
    print("="*60)
    
    summary = report['summary']
    print(f"Total de archivos: {summary['total_files']}")
    print(f"Tamaño total: {summary['total_size_gb']:.2f} GB")
    print(f"Score de seguridad promedio: {summary['avg_safety_score']:.3f}")
    
    print(f"\n📁 Categorías:")
    for category, count in report['categories'].items():
        print(f"  {category}: {count}")
    
    print(f"\n🎯 Recomendaciones:")
    for action, count in report['recommendations'].items():
        print(f"  {action}: {count}")
    
    cleanup = report['cleanup_potential']
    print(f"\n🧹 Potencial de limpieza:")
    print(f"  Archivos a eliminar: {cleanup['delete_candidates']}")
    print(f"  Espacio a liberar: {cleanup['delete_size_gb']:.2f} GB ({cleanup['space_savings_percent']:.1f}%)")
    
    # Mostrar resumen de clusters
    if classifier.is_trained:
        print(f"\n🔍 Clusters identificados:")
        cluster_summaries = classifier.get_cluster_summary(classifications)
        for cluster_id, summary in cluster_summaries.items():
            print(f"  Cluster {cluster_id}: {summary['total_files']} archivos, "
                  f"{summary['total_size_mb']:.1f} MB")
    
    # Guardar resultados
    success = classifier.save_results(classifications, args.output, args.format)
    
    if success:
        print(f"\n✅ Clasificación completada. Resultados guardados en: {args.output}")
    else:
        print(f"\n❌ Error guardando resultados")


if __name__ == "__main__":
    main()
    
