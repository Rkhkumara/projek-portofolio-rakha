import yaml
import os
import sys

def load_config(config_path: str = None) -> dict:
    """
    Memuat konfigurasi dari file YAML.
    Secara otomatis mencari config.yaml dari root proyek (folder tempat skrip dipanggil).
    """
    if config_path is None:
        # Cari config.yaml dari root proyek (kompatibel dijalankan dari manapun)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file tidak ditemukan di: {config_path}\n"
            "Pastikan Anda menjalankan script dari root folder proyek."
        )

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
