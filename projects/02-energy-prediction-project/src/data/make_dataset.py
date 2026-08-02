import logging
import os
import sys

# Pastikan root proyek ada di sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    config = load_config()
    logging.info("Tugas: Mengambil raw data (misal via SQL/API) dan simpan ke raw/")
    # TODO: Implementasikan logika pengambilan data asli
    # Contoh: download dari API cuaca, tarik dari database IoT, dsb.
    pass


if __name__ == "__main__":
    main()
