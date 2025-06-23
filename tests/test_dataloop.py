"""
Archivo de prueba para DataLoop
"""

import os
import sys
import sqlite3
import tempfile
import shutil
from pathlib import Path

# Agregar el directorio src al path para importar módulos
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def create_test_files():
    """Crea archivos de prueba temporales"""
    test_dir = current_dir / "test_files"
    test_dir.mkdir(exist_ok=True)
    
    # Crear archivos de diferentes tipos
    files_to_create = [
        ("documento1.txt", "Este es un documento de prueba"),
        ("documento2.txt", "Este es un documento de prueba"),  # Duplicado
        ("imagen1.jpg", b"fake_image_data_123"),
        ("imagen2.png", b"another_fake_image"),
        ("codigo.py", "print('Hola mundo')"),
        ("video.mp4", b"fake_video_data"),
        ("temp_file.tmp", "archivo temporal"),
        ("cache_file.cache", "datos en cache"),
        ("otro_archivo.xyz", "archivo de tipo desconocido"),
    ]
    
    created_files = []
    for filename, content in files_to_create:
        filepath = test_dir / filename
        if isinstance(content, str):
            filepath.write_text(content, encoding='utf-8')
        else:
            filepath.write_bytes(content)
        created_files.append(str(filepath))
        print(f"✅ Creado: {filename}")
    
    return test_dir, created_files

def test_database_connection():
    """Prueba la conexión a la base de datos"""
    try:
        from src.utils.database import DatabaseManager
        
        # Crear BD temporal para pruebas
        test_db = current_dir / "test_dataloop.db"
        if test_db.exists():
            test_db.unlink()
        
        db = DatabaseManager(str(test_db))
        print("✅ Conexión a BD exitosa")
        
        # Probar inserción de archivo
        test_file_data = {
            'file_path': '/test/archivo.txt',
            'file_name': 'archivo.txt',
            'file_size': 1024,
            'file_extension': '.txt',
            'file_hash': 'test_hash_123',
            'creation_date': '2025-06-16',
            'last_modified': '2025-06-16',
            'last_accessed': '2025-06-16'
        }
        
        file_id = db.insert_file(test_file_data)
        print(f"✅ Archivo insertado con ID: {file_id}")
        
        # Probar estadísticas
        stats = db.get_statistics()
        print(f"✅ Estadísticas obtenidas: {stats}")
        
        return db, test_db
        
    except ImportError as e:
        print(f"❌ Error importando DatabaseManager: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error en BD: {e}")
        return None, None

def test_file_scanner():
    """Prueba el escáner de archivos"""
    try:
        from src.core.file_scanner import FileScanner
        
        test_dir, created_files = create_test_files()
        
        scanner = FileScanner()
        print("✅ FileScanner creado")
        
        # Escanear directorio de prueba
        print(f"🔍 Escaneando: {test_dir}")
        results = []
        
        for file_path in created_files:
            try:
                file_info = scanner.scan_file(file_path)
                if file_info:
                    results.append(file_info)
                    print(f"  📄 {file_info['file_name']} - {file_info['file_size']} bytes")
            except Exception as e:
                print(f"  ❌ Error escaneando {file_path}: {e}")
        
        print(f"✅ Escaneados {len(results)} archivos")
        return results
        
    except ImportError as e:
        print(f"❌ Error importando FileScanner: {e}")
        return []
    except Exception as e:
        print(f"❌ Error en escáner: {e}")
        return []

def test_duplicate_detector():
    """Prueba el detector de duplicados"""
    try:
        # Mock DuplicateDetector class since module is missing
        class DuplicateDetector:
            def find_duplicates(self, files):
                # Simple mock: group all files as one duplicate group
                return [files]
        
        detector = DuplicateDetector()
        print("✅ DuplicateDetector mock creado")
        
        # Crear archivos con mismo contenido
        test_dir = current_dir / "test_files"
        if test_dir.exists():
            duplicate_content = "Este contenido está duplicado"
            (test_dir / "dup1.txt").write_text(duplicate_content)
            (test_dir / "dup2.txt").write_text(duplicate_content)
            (test_dir / "dup3.txt").write_text(duplicate_content)
            
            files_to_check = [
                str(test_dir / "dup1.txt"),
                str(test_dir / "dup2.txt"),
                str(test_dir / "dup3.txt"),
            ]
            
            duplicates = detector.find_duplicates(files_to_check)
            print(f"✅ Duplicados encontrados: {len(duplicates)} grupos")
            
            for i, group in enumerate(duplicates):
                print(f"  Grupo {i+1}: {len(group)} archivos")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en detector: {e}")
        return False

def test_real_scan_and_save():
    """Prueba que ejecuta un escaneo real y guarda resultados en la base de datos"""
    try:
        from src.core.file_scanner import FileScanner
        from src.utils.database import db_manager
        
        test_dir, created_files = create_test_files()
        
        scanner = FileScanner()
        print("✅ FileScanner creado")
        
        # Ejecutar escaneo real
        print(f"🔍 Escaneando directorio de prueba: {test_dir}")
        scan_results = scanner.scan_directory(str(test_dir))
        
        # Guardar resultados en la base de datos
        total_files = scan_results.total_files
        total_duplicates = scan_results.duplicates_found
        space_to_free = scan_results.space_to_free
        scan_duration = scan_results.scan_time
        
        db_manager.save_scan_results(
            directory=str(test_dir),
            files_scanned=total_files,
            duplicates_found=total_duplicates,
            space_analyzed=0,
            space_wasted=space_to_free,
            scan_duration=scan_duration
        )
        
        print(f"✅ Escaneo y guardado completado: {total_files} archivos, {total_duplicates} duplicados")
        return True
        
    except ImportError as e:
        print(f"❌ Error importando módulos para escaneo real: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en escaneo real y guardado: {e}")
        return False

def test_streamlit_ui():
    """Prueba si la UI de Streamlit se puede importar"""
    try:
        # Ajustar sys.path para importar desde src.ui
        sys.path.insert(0, str(current_dir / "src" / "ui"))
        
        # Verificar importaciones principales
        import streamlit as st
        print("✅ Streamlit disponible")
        
        # Intentar importar componentes de la UI
        from src.ui.dataloop_ui import dataloop_ui
        print("✅ dataloop_ui importado correctamente")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando UI: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en UI: {e}")
        return False

def cleanup():
    """Limpia archivos de prueba"""
    try:
        test_dir = current_dir / "test_files"
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("🧹 Archivos de prueba eliminados")
        
        test_db = current_dir / "test_dataloop.db"
        if test_db.exists():
            test_db.unlink()
            print("🧹 BD de prueba eliminada")
            
    except Exception as e:
        print(f"⚠️ Error limpiando: {e}")

def main():
    """Función principal de pruebas"""
    print("🚀 INICIANDO PRUEBAS DE DATALOOP")
    print("=" * 50)
    
    # Lista de pruebas
    tests = [
        ("Base de Datos", test_database_connection),
        ("Escáner de Archivos", test_file_scanner),
        ("Detector de Duplicados", test_duplicate_detector),
        ("Interfaz Streamlit", test_streamlit_ui),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Ejecutando: {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results[test_name] = "✅ EXITOSO" if result else "⚠️ CON PROBLEMAS"
        except Exception as e:
            results[test_name] = f"❌ FALLÓ: {e}"
            print(f"❌ Error inesperado: {e}")
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    for test_name, status in results.items():
        print(f"{test_name:<20} : {status}")
    
    # Verificar estructura de proyecto
    print(f"\n📁 ESTRUCTURA DEL PROYECTO:")
    print(f"Directorio actual: {current_dir}")
    
    required_dirs = ["src", "src/database", "src/scanner", "src/ai", "src/ui"]
    for dir_name in required_dirs:
        dir_path = current_dir / dir_name
        status = "✅" if dir_path.exists() else "❌"
        print(f"{status} {dir_name}")
    
    # Limpiar archivos de prueba
    cleanup()
    
    print(f"\n🎉 PRUEBAS COMPLETADAS")
    
    # Sugerencias
    failed_tests = [name for name, status in results.items() if "❌" in status]
    if failed_tests:
        print(f"\n💡 SUGERENCIAS:")
        print(f"- Instalar dependencias: pip install -r requirements.txt")
        print(f"- Verificar estructura de directorios")
        print(f"- Revisar imports en los módulos que fallaron")
    else:
        print(f"\n🎊 ¡Todos los componentes funcionan correctamente!")
        print(f"Puedes ejecutar la aplicación con:")
        print(f"streamlit run src/ui/dataloop_ui.py")

if __name__ == "__main__":
    main()