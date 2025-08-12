import os
import json
from pathlib import Path
from dotenv import load_dotenv

def main():
    load_dotenv()  # Carga las variables desde el .env

    print("=== Validación inicial de Vertex AI ===")

    # Leer variables del .env
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # 1) Verificar que existen en el .env
    if not project:
        raise ValueError("❌ Falta GCP_PROJECT en el .env")
    if not location:
        raise ValueError("❌ Falta GCP_LOCATION en el .env")
    if not creds_path:
        raise ValueError("❌ Falta GOOGLE_APPLICATION_CREDENTIALS en el .env")

    print(f"Proyecto: {project}")
    print(f"Ubicación: {location}")
    print(f"Ruta credenciales: {creds_path}")

    # 2) Verificar que el archivo existe
    creds_file = Path(creds_path)
    if not creds_file.exists():
        raise FileNotFoundError(f"❌ No se encontró el archivo de credenciales en: {creds_path}")

    # 3) Verificar que el JSON es válido y de tipo service_account
    try:
        with open(creds_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ El archivo no es un JSON válido: {e}")

    if data.get("type") != "service_account":
        raise ValueError("❌ El JSON no parece ser una cuenta de servicio (service_account)")

    print("✅ Variables y archivo de credenciales verificados correctamente.")

if __name__ == "__main__":
    main()
