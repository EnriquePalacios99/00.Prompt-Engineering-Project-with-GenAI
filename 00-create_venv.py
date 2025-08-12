import os
import sys
import subprocess
from pathlib import Path

def main():
    venv_path = Path(".venv")

    if venv_path.exists():
        print("⚠️ El entorno virtual .venv ya existe. No se creará de nuevo.")
        return

    print("=== Creando entorno virtual (.venv) ===")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
        print("✅ Entorno virtual creado correctamente en .venv")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creando el entorno virtual: {e}")
        sys.exit(1)

    print("\nPara activarlo:")
    if os.name == "nt":  # Windows
        print(r".venv\Scripts\activate")
    else:  # Linux / MacOS / Codespaces
        print("source .venv/bin/activate")

if __name__ == "__main__":
    main()
