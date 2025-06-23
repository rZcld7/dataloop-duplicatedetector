"""
DataLoop v3.0 - Detector de Archivos Duplicados
Nivel 3: Inteligencia & UX

Ejecutar con: streamlit run src/ui/dataloop_ui.py
"""

import sys
from pathlib import Path

# Agregar el directorio src al path para imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Import principal
from src.ui.dataloop_ui import main

if __name__ == "__main__":
    main()