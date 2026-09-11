"""Fail-closed scientific authority for the deterministic communication system.

The repository contains numerical communication-system simulation and optional
CuPy acceleration, not machine-learning optimization.  This audit prevents that
classification from becoming a permanent exemption: introducing PyTorch,
TensorFlow/JAX/sklearn training, backpropagation, or optimizer-step source causes a
hard failure until a real trainable DAG is added.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("digital_comm", "scripts", "app.py")
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pytorch", re.compile(r"\b(torch|pytorch|torchvision|torchaudio)\b", re.I)),
    ("tensorflow", re.compile(r"\b(tensorflow|keras|tflite)\b", re.I)),
    ("jax", re.compile(r"\b(jax|flax|optax|haiku)\b", re.I)),
    ("sklearn", re.compile(r"\b(scikit[- ]?learn|sklearn)\b", re.I)),
    ("optimizer", re.compile(r"\boptimizer\.(step|zero_grad)|\bAdamW?\s*\(|\bSGD\s*\(", re.I)),
    ("backprop", re.compile(r"\.backward\s*\(|backpropagation|gradient[_ -]?descent", re.I)),
    ("training-loop", re.compile(r"\b(train|training)[_ -]?(epoch|loop|step)|early[_ -]?stopping", re.I)),
)


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    category: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class Audit:
    files: tuple[str, ...]
    findings: tuple[Finding, ...]

    @property
    def complete(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "repository": "Anurag9000/End-to-End-Digital-Communication-System",
            "classification": "deterministic_signal_processing_simulation",
            "files": list(self.files),
            "findings": [asdict(row) for row in self.findings],
            "complete": self.complete,
            "optional_gpu_backend": "cupy-cuda12x",
            "trainable_surface_present": not self.complete,
            "source_configuration_only": True,
            "execution_claim_emitted": False,
        }


def _paths(root: Path):
    for name in SCAN_ROOTS:
        path = root / name
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.py"))


def audit(root: Path = ROOT) -> Audit:
    files: list[str] = []
    findings: list[Finding] = []
    for path in _paths(root):
        relative = path.relative_to(root).as_posix()
        files.append(relative)
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for category, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(Finding(relative, line_number, category, line.strip()[:240]))
    return Audit(tuple(files), tuple(findings))
