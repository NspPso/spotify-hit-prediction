from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spotify_hit_project.pipeline import run_pipeline


def main() -> None:
    context = run_pipeline()
    print(json.dumps(context, indent=2))


if __name__ == "__main__":
    main()
