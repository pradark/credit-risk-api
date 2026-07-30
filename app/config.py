from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        BASE_DIR / "models" / "model.pkl"
    )
)