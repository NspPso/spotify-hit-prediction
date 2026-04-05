from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spotify_hit_project.pipeline import download_data


def main() -> None:
    data_path = download_data(force=False)
    print(f"Saved dataset to {data_path}")


if __name__ == "__main__":
    main()
