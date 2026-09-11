#!/usr/bin/env python3
"""One-command exhaustive scientific controller for the digital communication repo.

This repository has no retained optimizer/model-training surface. Its scientific
work is deterministic end-to-end signal-processing simulation across source, PCM,
line coding/pulse shaping, channel, receiver, BER and parameter sweeps, with NumPy
and optional CuPy execution.  The central controller therefore executes the source
audit, tests, exhaustive parameter-grid verifier and report-generation pipeline
instead of fabricating ML jobs. A future training framework/optimizer marker makes
the authority fail closed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
REPOSITORY = "Anurag9000/End-to-End-Digital-Communication-System"
CONTROLLER_COMMIT = "fd34a95d18892df7fb14d1efbb99076a7810fb91"
CONTROLLER_BLOB = "05ef472b29933f18e956c69dfb7e543921ddaff5"
CONTROLLER_URL = (
    f"https://raw.githubusercontent.com/Anurag9000/RigorousRAG/{CONTROLLER_COMMIT}/"
    "tools/universal_training_controller_entry.py"
)
RESTART = {"exact_resume": True, "deterministic": True, "idempotent": True, "atomic_outputs": True}


def job(job_id: str, command: list[str], *, depends: list[str] | None = None, phase: str = "validation", device: bool = False, artifacts: list[str] | None = None) -> dict:
    return {
        "id": job_id,
        "command": command,
        "phase": phase,
        "family": "digital-communications/scientific-lifecycle",
        "device_capable": device,
        "depends_on": list(depends or []),
        "is_training_job": False,
        "resume_strategy": "restart_exact",
        "checkpoint_contract": dict(RESTART),
        "deterministic": True,
        "idempotent": True,
        "atomic_outputs": True,
        "early_stopping_applicable": False,
        "early_stopping_exception_reason": "non-optimizer deterministic simulation node",
        "completion_artifacts": list(artifacts or []),
    }


PROFILE = {
    "repository": REPOSITORY,
    "scientific_authority": "training_control/digital_comm_scientific_authority.py",
    "jobs": [
        job(
            "audit-no-trainable-surface",
            [sys.executable, "scripts/audit_scientific_authority.py"],
            phase="audit",
            artifacts=["artifacts/training_control/scientific_authority.json"],
        ),
        job(
            "test-complete-source",
            [sys.executable, "-m", "pytest", "-q"],
            depends=["audit-no-trainable-surface"],
            phase="test",
        ),
        job(
            "verify-complete-parameter-grid",
            [sys.executable, "scripts/verify_parameter_grid.py"],
            depends=["test-complete-source"],
            phase="experiment",
            device=True,
        ),
        job(
            "generate-complete-report",
            [sys.executable, "scripts/generate_report.py"],
            depends=["verify-complete-parameter-grid"],
            phase="reporting",
            artifacts=["report"],
        ),
    ],
    "preferred_training_entrypoints": [],
    "preferred_dataset_entrypoints": [],
    "dynamic_registry_covers": [
        "digital_comm/**/*.py",
        "app.py",
        "scripts/verify_parameter_grid.py",
        "scripts/generate_report.py",
        "requirements*.txt",
        "training_control/digital_comm_scientific_authority.py",
    ],
    "ignore_entrypoints": [
        "run_all_training.py",
        "scripts/audit_scientific_authority.py",
    ],
    "strict_coverage": True,
    "require_native_resume": True,
    "require_exact_resume": True,
    "require_training_exact_resume": True,
    "require_training_early_stopping": True,
    "require_well_formed_training_exemptions": True,
    "require_dag_enforcement": True,
    "require_model_surface_accounting": True,
    "require_workload_surface_accounting": True,
    "require_literal_opf_mechanism_parity": True,
    "require_registry_member_accounting": True,
    "require_dynamic_registry_accounting": True,
    "require_scientific_component_config_accounting": True,
    "require_declared_combination_accounting": True,
    "require_scientific_ontology_accounting": True,
    "require_declarative_scientific_source_accounting": True,
    "require_extended_scientific_component_accounting": True,
    "require_full_scientific_choice_accounting": True,
    "require_role_paradigm_protocol_accounting": True,
    "require_existing_job_targets": True,
    "require_source_proven_training_exact_resume": True,
    "require_source_proven_training_early_stopping": True,
    "require_all_retained_trainable_source_reachability": True,
    "auto_console_training_jobs": False,
    "auto_console_subcommand_jobs": False,
}


def blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main() -> int:
    cache = ROOT / ".training_control" / "universal_training_controller_entry.py"
    if not cache.is_file() or blob(cache.read_bytes()) != CONTROLLER_BLOB:
        data = urllib.request.urlopen(CONTROLLER_URL, timeout=60).read()
        actual = blob(data)
        if actual != CONTROLLER_BLOB:
            raise RuntimeError(f"Pinned controller checksum mismatch: {actual} != {CONTROLLER_BLOB}")
        atomic(cache, data)
    profile = ROOT / ".training_control" / "digital_comm_scientific_v37.json"
    atomic(profile, (json.dumps(PROFILE, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    env = os.environ.copy()
    env.pop("TRAINING_CONTROL_PROFILE", None)
    env["TRAINING_CONTROL_PROFILE_FILE"] = str(profile)
    env["TRAINING_CONTROL_REPO_ROOT"] = str(ROOT)
    env.setdefault("TRAINING_CONTROL_TERMINATION_GRACE_SEC", "30")
    return subprocess.call([sys.executable, str(cache), *sys.argv[1:]], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
