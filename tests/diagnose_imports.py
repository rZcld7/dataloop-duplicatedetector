"""
Script para diagnosticar importaciones circulares en el proyecto DataLoop
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque


class ImportAnalyzer:
    """Analizador de importaciones para detectar ciclos"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.imports_graph = defaultdict(set)
        self.file_imports = {}
        self.python_files = []
        
    def find_python_files(self) -> List[Path]:
        """Encuentra todos los archivos Python en el proyecto"""
        python_files = []
        
        # Buscar en src/ principalmente
        src_dir = self.project_root / "src"
        if src_dir.exists():
            for file_path in src_dir.rglob("*.py"):
                if not any(part.startswith('.') for part in file_path.parts):
                    python_files.append(file_path)
        
        # Buscar en la raíz también
        for file_path in self.project_root.glob("*.py"):
            python_files.append(file_path)
            
        self.python_files = python_files
        return python_files
    
    def extract_imports(self, file_path: Path) -> Set[str]:
        """Extrae las importaciones de un archivo Python"""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parsear el AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                print(f"⚠️  Error de sintaxis en {file_path}: {e}")
                return imports
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                        # También agregar importaciones específicas para detectar ciclos más granulares
                        for alias in node.names:
                            full_import = f"{node.module}.{alias.name}"
                            imports.add(full_import)
            
            return imports
            
        except Exception as e:
            print(f"❌ Error leyendo {file_path}: {e}")
            return imports
    
    def get_module_name(self, file_path: Path) -> str:
        """Convierte un path de archivo a nombre de módulo"""
        try:
            # Hacer el path relativo al proyecto
            rel_path = file_path.relative_to(self.project_root)
            
            # Convertir path a nombre de módulo
            parts = list(rel_path.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1].replace(".py", "")
            
            return ".".join(parts)
            
        except ValueError:
            # Si el archivo no está dentro del proyecto
            return str(file_path.stem)
    
    def build_import_graph(self):
        """Construye el grafo de importaciones"""
        print("🔍 Analizando importaciones...")
        
        files = self.find_python_files()
        print(f"📁 Encontrados {len(files)} archivos Python")
        
        for file_path in files:
            module_name = self.get_module_name(file_path)
            imports = self.extract_imports(file_path)
            
            self.file_imports[module_name] = {
                'file_path': file_path,
                'imports': imports
            }
            
            # Filtrar solo importaciones internas del proyecto
            internal_imports = set()
            for imp in imports:
                # Considerar importaciones que empiecen con 'src.' o sean módulos internos
                if (imp.startswith('src.') or 
                    imp.startswith('core.') or 
                    imp.startswith('utils.') or 
                    imp.startswith('ui.') or
                    imp in [self.get_module_name(f) for f in files]):
                    internal_imports.add(imp)
            
            self.imports_graph[module_name] = internal_imports
            
        print(f"📊 Construido grafo con {len(self.imports_graph)} módulos")
    
    def find_cycles(self) -> List[List[str]]:
        """Encuentra ciclos en el grafo de importaciones usando DFS"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Encontramos un ciclo
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # Visitar dependencias
            for dependency in self.imports_graph.get(node, set()):
                # Normalizar el nombre del módulo para la comparación
                dep_normalized = self._normalize_module_name(dependency)
                if dep_normalized in self.imports_graph:
                    dfs(dep_normalized, path.copy())
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        # Ejecutar DFS desde cada nodo no visitado
        for module in self.imports_graph:
            if module not in visited:
                dfs(module, [])
        
        return cycles
    
    def _normalize_module_name(self, module_name: str) -> str:
        """Normaliza nombres de módulos para comparación"""
        # Remover partes específicas de importaciones (ej: 'src.utils.config.Config' -> 'src.utils.config')
        parts = module_name.split('.')
        
        # Si es una importación específica (última parte empieza con mayúscula), quitar la última parte
        if len(parts) > 1 and parts[-1][0].isupper():
            return '.'.join(parts[:-1])
        
        return module_name
    
    def find_simple_cycles(self) -> List[Tuple[str, str]]:
        """Encuentra ciclos simples (A->B->A)"""
        simple_cycles = []
        
        for module_a in self.imports_graph:
            for dep_b in self.imports_graph[module_a]:
                dep_b_norm = self._normalize_module_name(dep_b)
                if dep_b_norm in self.imports_graph:
                    for dep_c in self.imports_graph[dep_b_norm]:
                        dep_c_norm = self._normalize_module_name(dep_c)
                        if dep_c_norm == module_a:
                            simple_cycles.append((module_a, dep_b_norm))
        
        return simple_cycles
    
    def analyze_config_dependencies(self) -> Dict[str, List[str]]:
        """Analiza específicamente las dependencias de Config"""
        config_deps = {}
        
        for module, data in self.file_imports.items():
            if 'config' in module.lower():
                imports = [imp for imp in data['imports'] if imp.startswith('src.')]
                config_deps[module] = imports
                
        return config_deps
    
    def find_who_imports_config(self) -> List[str]:
        """Encuentra qué módulos importan Config"""
        importers = []
        
        for module, data in self.file_imports.items():
            imports = data['imports']
            for imp in imports:
                if 'config' in imp.lower() and 'Config' in imp:
                    importers.append(module)
                    break
        
        return importers
    
    def print_detailed_analysis(self):
        """Imprime un análisis detallado"""
        print("\n" + "="*80)
        print("🔍 ANÁLISIS DETALLADO DE IMPORTACIONES")
        print("="*80)
        
        # Análisis general
        print(f"\n📊 ESTADÍSTICAS GENERALES:")
        print(f"   • Archivos Python: {len(self.python_files)}")
        print(f"   • Módulos con importaciones: {len(self.imports_graph)}")
        
        # Análisis de Config específicamente
        print(f"\n🔧 ANÁLISIS DE CONFIG:")
        config_deps = self.analyze_config_dependencies()
        if config_deps:
            for module, imports in config_deps.items():
                print(f"   • {module}:")
                print(f"     📁 Archivo: {self.file_imports[module]['file_path']}")
                if imports:
                    print(f"     📥 Importa: {', '.join(imports)}")
                else:
                    print(f"     📥 No importa módulos internos")
        
        # Quién importa Config
        print(f"\n🎯 MÓDULOS QUE IMPORTAN CONFIG:")
        config_importers = self.find_who_imports_config()
        if config_importers:
            for importer in config_importers:
                print(f"   • {importer}")
                print(f"     📁 {self.file_imports[importer]['file_path']}")
        else:
            print("   ✅ Ningún módulo importa Config directamente")
        
        # Buscar ciclos simples
        print(f"\n🔄 CICLOS SIMPLES DETECTADOS:")
        simple_cycles = self.find_simple_cycles()
        if simple_cycles:
            for cycle in simple_cycles:
                print(f"   ❌ {cycle[0]} ↔ {cycle[1]}")
        else:
            print("   ✅ No se detectaron ciclos simples")
        
        # Buscar todos los ciclos
        print(f"\n🌀 TODOS LOS CICLOS:")
        all_cycles = self.find_cycles()
        if all_cycles:
            for i, cycle in enumerate(all_cycles, 1):
                print(f"   ❌ Ciclo {i}: {' → '.join(cycle)}")
        else:
            print("   ✅ No se detectaron ciclos")
        
        # Mostrar grafo de importaciones para debug
        print(f"\n🕸️ GRAFO DE IMPORTACIONES (solo módulos internos):")
        for module, imports in self.imports_graph.items():
            if imports:
                print(f"   📦 {module}:")
                for imp in sorted(imports):
                    print(f"      └─ {imp}")
    
    def generate_fix_suggestions(self) -> List[str]:
        """Genera sugerencias para arreglar los ciclos"""
        suggestions = []
        
        # Sugerencias generales
        suggestions.append("🔧 SUGERENCIAS PARA ARREGLAR CICLOS:")
        suggestions.append("")
        
        config_deps = self.analyze_config_dependencies()
        config_importers = self.find_who_imports_config()
        
        if config_importers:
            suggestions.append("1. 📝 ELIMINAR IMPORTACIONES DE CONFIG:")
            for importer in config_importers:
                suggestions.append(f"   • En {importer}: remover 'from src.utils.config import Config'")
            suggestions.append("")
        
        if any(config_deps.values()):
            suggestions.append("2. 🚫 ELIMINAR IMPORTACIONES EN CONFIG:")
            for module, imports in config_deps.items():
                if imports:
                    suggestions.append(f"   • En {module}: remover importaciones de {', '.join(imports)}")
            suggestions.append("")
        
        suggestions.append("3. 🏗️ REESTRUCTURAR CÓDIGO:")
        suggestions.append("   • Mover lógica que depende de otros módulos fuera de Config")
        suggestions.append("   • Usar lazy loading para dependencias")
        suggestions.append("   • Crear funciones factory en lugar de importaciones directas")
        suggestions.append("")
        
        suggestions.append("4. 📋 PATRÓN RECOMENDADO:")
        suggestions.append("   • Config debe ser independiente (no importar nada del proyecto)")
        suggestions.append("   • Otros módulos pueden importar Config")
        suggestions.append("   • Usar métodos get/set para acceso dinámico")
        
        return suggestions


def main():
    """Función principal del diagnóstico"""
    print("🚀 DIAGNÓSTICO DE IMPORTACIONES CIRCULARES - DataLoop")
    print("="*60)
    
    # Detectar directorio del proyecto
    current_dir = Path.cwd()
    project_root = None
    
    # Buscar hacia arriba hasta encontrar src/ o .git/
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "src").exists() or (parent / ".git").exists():
            project_root = parent
            break
    
    if not project_root:
        project_root = current_dir
        print(f"⚠️  No se encontró raíz del proyecto, usando: {project_root}")
    else:
        print(f"📁 Raíz del proyecto detectada: {project_root}")
    
    # Crear analizador
    analyzer = ImportAnalyzer(str(project_root))
    
    try:
        # Construir grafo de importaciones
        analyzer.build_import_graph()
        
        # Realizar análisis detallado
        analyzer.print_detailed_analysis()
        
        # Generar sugerencias
        print("\n" + "="*80)
        suggestions = analyzer.generate_fix_suggestions()
        for suggestion in suggestions:
            print(suggestion)
        
        print("\n" + "="*80)
        print("✅ Análisis completado.")
        print("💡 Tip: Copia la salida para analizar los ciclos detectados")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    