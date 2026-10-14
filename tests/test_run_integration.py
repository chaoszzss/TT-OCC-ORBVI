"""Real first-frame pipeline checks; these never satisfy full-bag acceptance."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_run", PROJECT / "scripts" / "verify_run.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture(scope="module")
def real_limited_run(tmp_path_factory):
    prepared = PROJECT / "prepared"
    if not (prepared / "trajectory.npz").is_file():
        pytest.skip("real prepared full-bag trajectory not available")
    from orbvi_occ.run import run
    temporary = tmp_path_factory.mktemp("real_run_integration")
    config = json.loads((PROJECT / "configs" / "lidar.json").read_text())
    config["prepared"] = str(prepared)
    config["semantics"]["enabled"] = False
    config["output"] = str(temporary / "output")
    config_path = temporary / "config.json"
    config_path.write_text(json.dumps(config))
    summary = run(config_path, limit=1)
    return temporary / "output", config_path, summary


def test_real_pipeline_saves_consistent_source_and_tt_geometry(real_limited_run):
    output, _, summary = real_limited_run
    assert summary["complete"] is False
    saved_summary = json.loads((output / "summary.json").read_text())
    assert "acceptance" in saved_summary, "plot metadata overwrote final summary.json"
    assert saved_summary["acceptance"]["recorded_frames"] == 1
    assert (output / "timeline.png").is_file()
    assert (output / "timeline.json").is_file()
    rows = [json.loads(r) for r in (output / "frames.jsonl").read_text().splitlines()]
    assert rows[0]["index"] == 0
    assert rows[0]["raw_source_points"] == 19968
    with np.load(output / "occupancy" / "000000.npz", allow_pickle=False) as occ:
        assert int(occ["stamp_ns"]) == rows[0]["stamp_ns"]
        assert len(occ["fused_indices"]) == rows[0]["fused_occupied_voxels"]
        assert np.all(occ["fused_labels"] == 255)
        assert np.all(np.isfinite(occ["T_world_sensor"]))
    with np.load(output / "traversability" / "000000.npz", allow_pickle=False) as grid:
        assert not np.any(grid["observed_free"] & grid["collision3d"])
        assert not np.any(grid["collision3d"] & ~grid["occupied3d"])
        assert np.isfinite(grid["clearance_coverage"]).all()


def test_independent_verifier_rejects_fragment_with_valid_artifacts(real_limited_run):
    output, _, _ = real_limited_run
    report = verifier.verify(output)
    assert not report["accepted"]
    failures = {c["name"] for c in report["checks"] if not c["passed"]}
    assert "all_recording_frames_logged" in failures
    assert "complete_occupancy_artifact_set" in failures
    assert "full_run_not_development_limit" in failures
    assert report["statistics"]["valid_occupancy_artifacts"] == 1
    assert report["statistics"]["valid_traversability_artifacts"] == 1


def test_verifier_rejects_corrupted_voxel_indices(real_limited_run, tmp_path):
    import shutil
    output, _, _ = real_limited_run
    broken = tmp_path / "corrupted"
    shutil.copytree(output, broken)
    path = broken / "occupancy" / "000000.npz"
    with np.load(path, allow_pickle=False) as d:
        payload = {k: d[k].copy() for k in d.files}
    payload["fused_indices"][0, 0] = -1
    np.savez_compressed(path, **payload)
    report = verifier.verify(broken)
    assert not report["accepted"]
    assert report["statistics"]["artifact_errors"] >= 1
    assert any("out of bounds" in r["error"] for r in report["artifact_errors"])


def test_lidar_resume_rebuilds_two_frame_causal_prefix(real_limited_run):
    output, config, _ = real_limited_run
    from orbvi_occ.run import run
    # Simulate interruption after a metric commit but before its frame commit.
    metric_path=output/'heldout_metrics.jsonl'
    initial=[json.loads(line) for line in metric_path.read_text().splitlines()]
    orphan={**initial[0],'index':1}
    with metric_path.open('a') as journal:
        journal.write(json.dumps(orphan)+'\n')
    resumed = run(config, limit=2, resume=True)
    assert resumed["complete"] is False
    rows = [json.loads(r) for r in (output / "frames.jsonl").read_text().splitlines()]
    assert [r["index"] for r in rows] == [0, 1]
    assert rows[1]["algorithm"]["past_frames"] == 1
    assert (output / "occupancy" / "000001.npz").is_file()
    assert (output / "traversability" / "000001.npz").is_file()
    assert resumed["source_lidar_points_accounted"] == 39936
    assert resumed["heldout_geometry"]["frames"] == len(resumed["heldout_metrics"])
    assert any(m["index"] == 0 for m in resumed["heldout_metrics"])
    persisted=[json.loads(line) for line in metric_path.read_text().splitlines()]
    assert [m['index'] for m in persisted]==[m['index'] for m in resumed['heldout_metrics']]
    assert len({m['index'] for m in persisted})==len(persisted)


def test_verifier_cli_exits_nonzero_for_fragment(real_limited_run, tmp_path):
    import subprocess
    import sys
    output, _, _ = real_limited_run
    report_path = tmp_path / "independent_verification.json"
    result = subprocess.run([sys.executable, str(PROJECT / "scripts" / "verify_run.py"),
                             "--run", str(output), "--report", str(report_path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2
    report = json.loads(report_path.read_text())
    assert report["accepted"] is False
    assert report_path.with_suffix(".md").is_file()
