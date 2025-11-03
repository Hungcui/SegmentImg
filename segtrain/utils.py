from pathlib import Path
import random, numpy as np, tensorflow as tf
from typing import Optional

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def guess_google_drive_windows() -> Optional[Path]:
    # Try common streamed letters
    for letter in "GHIJKLMNOPQRSTUVWXYZ":
        p = Path(f"{letter}:/My Drive")
        if p.exists():
            return p
    # Try common mirrored folders
    candidates = [
        Path.home() / "My Drive",
        Path.home() / "Google Drive/My Drive",
        Path.home() / "GoogleDrive/My Drive",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None
