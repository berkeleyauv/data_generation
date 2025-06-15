from dotenv import load_dotenv
import os

load_dotenv()

BACKGROUNDS_DIR = os.getenv("BACKGROUNDS_DIR")
TARGET_ASSETS_DIR = os.getenv("TARGET_ASSETS_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# Safety check
assert all([BACKGROUNDS_DIR, TARGET_ASSETS_DIR, OUTPUT_DIR]), "Missing paths in .env"
