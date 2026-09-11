#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_control.digital_comm_scientific_authority import audit  # noqa: E402


def main() -> int:
    result = audit(ROOT)
    output = ROOT / "artifacts" / "training_control" / "scientific_authority.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
