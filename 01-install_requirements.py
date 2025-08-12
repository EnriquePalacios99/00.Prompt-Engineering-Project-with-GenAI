import subprocess
import sys
from pathlib import Path

def main():
    req_file = Path("requirements.txt")

    if not req_file.exists():
        print("❌ No se encontró requirements.txt en el directorio actual.")
        sys.exit(1)

    print(f"=== Instalando dependencias desde {req_file} ===")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        print("\n✅ Todas las dependencias de requirements.txt se instalaron correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
