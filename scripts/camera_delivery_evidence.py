"""Match published camera artifacts to a particular independent full-run audit.

Reading metadata and file hashes never launches neural inference. Directory names
alone are deliberately insufficient because a failed run can be rebuilt there.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_identity(run, metadata):
    run = Path(run).resolve()
    return {
        "config_sha256": metadata.get("config_sha256"),
        "bag_sha256": metadata.get("bag_sha256"),
        "inference_cache_key": metadata.get("inference_cache_key"),
        "run_json_sha256": file_sha256(run / "run.json"),
        "frames_jsonl_sha256": file_sha256(run / "frames.jsonl"),
    }


def matching_run_identity(qa, run, metadata):
    """Compare an independent report's exact metadata/journal byte snapshot."""
    recorded = qa.get("run_identity")
    if not isinstance(recorded, dict):
        return False
    try:
        current = run_identity(run, metadata)
    except OSError:
        return False
    return all(isinstance(value, str) and value and recorded.get(key) == value
               for key, value in current.items())


def matching_full_qa(qa, run, metadata, expected_frames):
    """A previous audit in the same pathname cannot approve replacement data."""
    return bool(
        qa.get("run_directory") == str(Path(run).resolve())
        and qa.get("accepted") is True
        and qa.get("allow_partial") is False
        and qa.get("expected_full_recording_frames") == expected_frames
        and qa.get("statistics", {}).get("verified_frames") == expected_frames
        and matching_run_identity(qa, run, metadata)
    )


def matching_residency_validation(report, metadata):
    """Distinguish old two-model probes from the actual three-stage execution."""
    cache_key = metadata.get("inference_cache_key")
    if not (report.get("accepted") is True and
            isinstance(cache_key, str) and cache_key and
            report.get("cache_key") == report.get("original_cache_key") == cache_key):
        return False
    provenance = metadata.get("input_provenance_sha256", {})
    main = "orbvi_occ/camera_semantic_run.py"
    execution = report.get("execution_source_sha256", {})
    recorded_main = execution.get(main, report.get("runtime_source_sha256"))
    if not provenance.get(main) or recorded_main != provenance[main]:
        return False
    if metadata.get("openseed_worker_entry"):
        required = (main, "orbvi_occ/openseed_residency.py", "scripts/openseed_resident_python.sh")
        return all(provenance.get(name) and execution.get(name) == provenance[name] for name in required)
    return True
