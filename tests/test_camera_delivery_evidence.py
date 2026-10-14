"""Published success must be bound to the exact audited recording artifacts."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from camera_delivery_evidence import matching_full_qa, matching_residency_validation, run_identity


def test_same_directory_rebuild_cannot_reuse_previously_accepted_qa(tmp_path):
    metadata = {"config_sha256": "configuration-one", "bag_sha256": "bag-one",
                "inference_cache_key": "frozen-inference-one"}
    (tmp_path / "run.json").write_text(json.dumps(metadata))
    (tmp_path / "frames.jsonl").write_text('{"index":0,"stamp_ns":42}\n')
    qa = {"run_directory": str(tmp_path.resolve()), "accepted": True,
          "allow_partial": False, "expected_full_recording_frames": 1,
          "statistics": {"verified_frames": 1}, "run_identity": run_identity(tmp_path, metadata)}
    assert matching_full_qa(qa, tmp_path, metadata, 1)
    # Same path, same number of committed rows, and same NN cache are insufficient.
    (tmp_path / "frames.jsonl").write_text('{"index":0,"stamp_ns":43}\n')
    assert not matching_full_qa(qa, tmp_path, metadata, 1)
    qa["run_identity"] = run_identity(tmp_path, metadata)
    assert matching_full_qa(qa, tmp_path, metadata, 1)
    (tmp_path / "run.json").write_text(json.dumps(metadata) + "\n")
    assert not matching_full_qa(qa, tmp_path, metadata, 1)


def test_partial_or_unidentified_qa_cannot_approve_full_delivery(tmp_path):
    metadata = {"config_sha256": "config", "bag_sha256": "bag", "inference_cache_key": "cache"}
    (tmp_path / "run.json").write_text(json.dumps(metadata))
    (tmp_path / "frames.jsonl").write_text('{}\n')
    qa = {"run_directory": str(tmp_path.resolve()), "accepted": True,
          "allow_partial": True, "expected_full_recording_frames": 2348,
          "statistics": {"verified_frames": 2348}, "run_identity": run_identity(tmp_path, metadata)}
    assert not matching_full_qa(qa, tmp_path, metadata, 2348)
    qa["allow_partial"] = False
    assert matching_full_qa(qa, tmp_path, metadata, 2348)
    qa.pop("run_identity")
    assert not matching_full_qa(qa, tmp_path, metadata, 2348)


def test_residency_proof_must_match_worker_and_launcher_not_only_main():
    main,worker,launcher = ('orbvi_occ/camera_semantic_run.py','orbvi_occ/openseed_residency.py',
                           'scripts/openseed_resident_python.sh')
    metadata = {'inference_cache_key':'fixed-perception','openseed_worker_entry':'resident launcher',
                'input_provenance_sha256':{main:'same-main',worker:'new-worker',launcher:'launcher'}}
    old = {'accepted':True,'cache_key':'fixed-perception','original_cache_key':'fixed-perception',
           'runtime_source_sha256':'same-main'}
    assert not matching_residency_validation(old,metadata), 'Missing worker provenance cannot approve the new GPU schedule'
    old['execution_source_sha256'] = {main:'same-main',worker:'old-worker',launcher:'launcher'}
    assert not matching_residency_validation(old,metadata), 'A reused main source does not make different workers equivalent'
    old['execution_source_sha256'][worker] = 'new-worker'
    assert matching_residency_validation(old,metadata)
