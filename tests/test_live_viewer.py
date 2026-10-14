"""Viewer checks use persisted artifacts and never launch model inference."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_viewer", PROJECT / "scripts" / "live_viewer.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


@contextmanager
def running_server(run, page=None):
    server = viewer.make_server(run, port=0, **({"page_path": page} if page else {}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(base, endpoint="/api/frame"):
    try:
        response = urlopen(base + endpoint, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.loads(response.read())


def write_frame(run, index=0, commit=True):
    """Small real-format archives let tests exercise interruption boundaries."""
    for name in ("occupancy", "traversability"):
        (run / name).mkdir(parents=True, exist_ok=True)
    bounds = np.array([[-1., -1., -1.], [1., 1., 1.]])
    stamp = 1_720_198_192_900_208_445 + index * 100_000_000
    current = np.array([[0, 0, 0]], dtype=np.int32)
    fused = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int32)
    np.savez_compressed(run / "occupancy" / f"{index:06d}.npz", bounds=bounds,
                        stamp_ns=stamp, voxel_size=1., current_indices=current,
                        fused_indices=fused, current_labels=np.array([255], dtype=np.uint8),
                        fused_labels=np.array([255, 20], dtype=np.uint8))
    np.savez_compressed(run / "traversability" / f"{index:06d}.npz", bounds=bounds.ravel(),
                        resolution=1., timestamp=stamp / 1e9,
                        traversability=np.array([[0, 1], [2, 3]], dtype=np.uint8))
    (run / "run.json").write_text(json.dumps({"route": "lidar", "expected_full_recording_frames": 2}))
    row = {"index": index, "stamp_ns": stamp, "current_occupied_voxels": 1,
           "fused_occupied_voxels": 2, "new_voxels_from_history": 1}
    if commit:
        with (run / "frames.jsonl").open("a") as journal:
            journal.write(json.dumps(row) + "\n")
    return row


def test_viewer_starts_before_run_then_updates_only_after_complete_commit(tmp_path):
    run = tmp_path / "future_run"
    with running_server(run) as base:
        status, waiting = request(base)
        assert status == 200 and waiting["status"] == "waiting"
        assert waiting["processed_frames"] == 0
        assert not run.exists(), "a read-only viewer created the inference output"
        row = write_frame(run, commit=False)
        assert request(base)[1]["status"] == "waiting", "NPZ alone is not a frame commit"
        serialized = json.dumps(row)
        (run / "frames.jsonl").write_text(serialized[:24])
        assert request(base)[1]["status"] == "waiting"
        with (run / "frames.jsonl").open("a") as journal:
            journal.write(serialized[24:])
        assert request(base)[1]["status"] == "waiting", "missing newline is uncommitted"
        with (run / "frames.jsonl").open("a") as journal:
            journal.write("\n")
        status, payload = request(base)
        assert status == 200 and payload["status"] == "ready"
        assert payload["index"] == 0 and payload["processed_frames"] == 1
        assert payload["centers"] == [[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5]]
        assert payload["labels"] == [255, 20]
        assert payload["grid"]["traversability"] == [0, 1, 2, 3]


def test_latest_never_selects_uncommitted_larger_npz(tmp_path):
    run = tmp_path / "run"
    write_frame(run, 0)
    row = write_frame(run, 1, commit=False)
    reader = viewer.RunReader(run)
    assert reader.frame()[0]["index"] == 0
    with (run / "frames.jsonl").open("a") as journal:
        journal.write(json.dumps(row))
    assert reader.frame()[0]["index"] == 0
    with (run / "frames.jsonl").open("a") as journal:
        journal.write("\n")
    assert reader.frame()[0]["index"] == 1
    assert reader.frame(0, "current")[0]["labels"] == [255]


def test_missing_or_half_written_artifacts_wait_and_recover(tmp_path):
    run = tmp_path / "run"
    row = write_frame(run)
    archive = run / "occupancy" / "000000.npz"
    valid = archive.read_bytes()
    archive.unlink()
    reader = viewer.RunReader(run)
    assert reader.frame()[0]["status"] == "waiting"
    archive.write_bytes(valid[:30])
    assert reader.frame()[0]["status"] == "waiting"
    archive.write_bytes(valid)
    payload, code = reader.frame()
    assert code == 200 and payload["status"] == "ready"
    assert payload["stamp_ns"] == row["stamp_ns"]
    assert payload["stamp_ns_text"] == str(row["stamp_ns"])


def test_complete_requires_valid_complete_summary_and_all_commits(tmp_path):
    run = tmp_path / "run"
    write_frame(run)
    (run / "summary.json").write_text('{"complete":true}')
    reader = viewer.RunReader(run)
    assert not reader.frame()[0]["complete"]
    write_frame(run, 1)
    assert reader.frame()[0]["complete"]
    (run / "summary.json").write_text('{"complete":')
    assert not reader.frame()[0]["complete"]
    (run / "summary.json").write_text('{"complete":false}')
    assert not reader.frame()[0]["complete"]


def test_expected_count_with_missing_time_step_is_not_complete(tmp_path):
    run = tmp_path / "run"
    write_frame(run, 0)
    write_frame(run, 2)
    (run / "summary.json").write_text('{"complete":true}')
    payload, code = viewer.RunReader(run).frame()
    assert code == 200 and payload["status"] == "ready"
    assert payload["processed_frames"] == payload["expected_frames"] == 2
    assert not payload["complete"], "Matching row count cannot hide the missing frame 1"


def test_truncated_journal_resets_cache_and_bad_committed_json_reports_error(tmp_path):
    run = tmp_path / "run"
    write_frame(run)
    write_frame(run, 1)
    reader = viewer.RunReader(run)
    assert reader.frame()[0]["index"] == 1
    row0 = json.loads((run / "frames.jsonl").read_text().splitlines()[0])
    (run / "frames.jsonl").write_text(json.dumps(row0) + "\n")
    assert reader.frame()[0]["processed_frames"] == 1
    with (run / "frames.jsonl").open("a") as journal:
        journal.write("{broken}\n")
    payload, code = reader.frame()
    assert code == 500 and payload["status"] == "error"


@pytest.mark.parametrize("endpoint", ["/api/frame?index=-1", "/api/frame?index=1.5",
                                        "/api/frame?mode=raw", "/api/frame?index=",
                                        "/api/frame?index=0&index=1", "/api/frame?path=/etc/passwd"])
def test_invalid_query_returns_400(tmp_path, endpoint):
    with running_server(tmp_path / "absent") as base:
        code, payload = request(base, endpoint)
        assert code == 400 and payload["status"] == "error"


@pytest.mark.parametrize("endpoint", ["/../README.md", "/%2e%2e/README.md", "/runs/lidar_full/run.json", "/api/frame/../run.json"])
def test_viewer_does_not_expose_workspace_files(tmp_path, endpoint):
    with running_server(tmp_path / "absent") as base:
        assert request(base, endpoint)[0] == 404


def test_homepage_and_camera_replay_use_the_same_server(tmp_path):
    run = tmp_path / "run"
    write_frame(run)
    (run / "run.json").write_text(json.dumps({"route": "camera", "expected_full_recording_frames": 2}))
    page = tmp_path / "index.html"
    page.write_text("<!doctype html><title>Local viewer</title>")
    with running_server(run, page) as base:
        with urlopen(base, timeout=5) as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert b"Local viewer" in response.read()
        code, payload = request(base, "/api/frame?index=0&mode=current")
        assert code == 200 and payload["route"] == "camera"
        assert len(payload["centers"]) == payload["current_count"] == 1


@pytest.mark.parametrize("route", ["lidar_full", "camera_full"])
def test_real_saved_npz_is_returned_exactly_over_http(route):
    run = PROJECT / "runs" / route
    if not (run / "frames.jsonl").is_file():
        pytest.skip("full-bag saved artifacts unavailable")
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    with running_server(run) as base:
        for index, mode in ((0, "current"), (rows[-1]["index"], "fused")):
            code, payload = request(base, f"/api/frame?index={index}&mode={mode}")
            assert code == 200 and payload["status"] == "ready"
            with np.load(run / "occupancy" / f"{index:06d}.npz", allow_pickle=False) as stored:
                expected = stored["bounds"][0] + (stored[f"{mode}_indices"].astype(float) + 0.5) * float(stored["voxel_size"])
                np.testing.assert_allclose(payload["centers"], expected, rtol=0, atol=1e-12)
                assert payload["labels"] == stored[f"{mode}_labels"].tolist()
                assert payload["stamp_ns"] == int(stored["stamp_ns"])
                assert payload["stamp_ns_text"] == str(int(stored["stamp_ns"]))
            with np.load(run / "traversability" / f"{index:06d}.npz", allow_pickle=False) as stored:
                assert payload["grid"]["shape"] == list(stored["traversability"].shape)
                assert payload["grid"]["traversability"] == stored["traversability"].ravel(order="C").tolist()
            assert payload["processed_frames"] == len(rows)
            assert payload["complete"] is True


def test_nonlocal_bind_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="localhost"):
        viewer.make_server(tmp_path, host="0.0.0.0", port=0)


def test_changed_artifact_invalidates_cached_payload_and_corruption_is_explicit(tmp_path):
    run = tmp_path / "run"
    write_frame(run)
    archive = run / "occupancy" / "000000.npz"
    with running_server(run) as base:
        assert request(base)[1]["labels"] == [255, 20]
        with np.load(archive, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        arrays["fused_labels"][1] = 42
        np.savez_compressed(archive, **arrays)
        assert request(base)[1]["labels"] == [255, 42]
        arrays["fused_indices"][0, 0] = -1
        np.savez_compressed(archive, **arrays)
        code, payload = request(base)
        assert code == 500 and payload["status"] == "error"
        assert "outside" in payload["status_detail"]


def test_crc_directory_invalidates_equal_size_archive_with_identical_stat_metadata(tmp_path,monkeypatch):
    """Deterministic timestamp collision; no sleep or weakened freshness check."""
    run=tmp_path/'run'
    write_frame(run)
    archive=run/'occupancy/000000.npz'
    original_fingerprint=viewer._fingerprint
    frozen={path:original_fingerprint(path) for path in (archive,run/'traversability/000000.npz')}
    monkeypatch.setattr(viewer,'_fingerprint',lambda path:frozen.get(path,original_fingerprint(path)))
    reader=viewer.RunReader(run)
    assert reader.frame()[0]['labels']==[255,20]
    old_size=archive.stat().st_size
    with np.load(archive,allow_pickle=False) as stored:
        arrays={name:stored[name].copy() for name in stored.files}
    arrays['fused_labels'][1]=42
    np.savez_compressed(archive,**arrays)
    assert archive.stat().st_size==old_size,'Fixture must reproduce the same-size rewrite'
    assert viewer._fingerprint(archive)==frozen[archive]
    payload,code=reader.frame()
    assert code==200 and payload['labels']==[255,42]


def test_corrupt_replacement_never_reuses_ready_cache_even_with_identical_stat_metadata(tmp_path,monkeypatch):
    run=tmp_path/'run'
    write_frame(run)
    archive=run/'occupancy/000000.npz'
    original_fingerprint=viewer._fingerprint
    frozen=original_fingerprint(archive)
    reader=viewer.RunReader(run)
    assert reader.frame()[0]['labels']==[255,20]
    monkeypatch.setattr(viewer,'_fingerprint',lambda path:frozen if path==archive else original_fingerprint(path))
    valid=archive.read_bytes()
    archive.write_bytes(valid[:-22])  # Remove end-of-central-directory; explicit corruption.
    payload,code=reader.frame()
    assert code==500 and payload['status']=='error'
    assert 'ZIP directory' in payload['status_detail']
    assert 'labels' not in payload,'A broken archive must not return a stale ready payload'
    archive.write_bytes(valid)
    payload,code=reader.frame()
    assert code==200 and payload['labels']==[255,20]
