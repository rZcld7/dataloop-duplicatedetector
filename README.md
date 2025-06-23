# DataLoop v3.0 - Detector de Archivos Duplicados

## Descripción
DataLoop es una aplicación que detecta archivos duplicados utilizando técnicas inteligentes y con una interfaz de usuario amigable. Esta herramienta te ayudará a identificar y gestionar archivos duplicados en tu sistema de manera eficiente.

## Requisitos previos
- Tener instalado Python 3.8 o superior.
- Tener acceso a una terminal o consola de comandos.
- (Opcional) Se recomienda usar un entorno virtual para aislar las dependencias del proyecto.

## Instalación

1. Clona o descarga este repositorio en tu equipo.

2. (Opcional) Crea y activa un entorno virtual:

   - En Windows:
     ```bash
     python -m venv EntornoV-dataloop
     EntornoV-dataloop\Scripts\activate.bat
     ```

   - En Linux/Mac:
     ```bash
     python3 -m venv EntornoV-dataloop
     source EntornoV-dataloop/bin/activate
     ```

3. Instala las dependencias necesarias con pip:

   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

Para iniciar la aplicación, ejecuta el siguiente comando en la terminal desde la raíz del proyecto:

```bash
streamlit run src/ui/dataloop_ui.py
```

Esto iniciará un servidor local y abrirá la aplicación en tu navegador web. Normalmente, la URL será:

```
http://localhost:8501
```

Si no se abre automáticamente, copia y pega esta dirección en tu navegador.

## Uso

- Una vez abierta la aplicación en el navegador, sigue las instrucciones en pantalla para cargar y analizar tus archivos.
- La aplicación detectará archivos duplicados y te permitirá gestionarlos de forma sencilla.

## Soporte

Si tienes alguna duda o problema, revisa los archivos de logs en la carpeta `logs/` para más detalles.

---

Este README proporciona las instrucciones básicas para instalar, ejecutar y usar DataLoop. Para más detalles, consulta la documentación interna o contacta con el desarrollador.
