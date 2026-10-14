#!/usr/bin/env bash
set -euo pipefail

TT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TT_RUNTIME_PYTHON="${TT_PROJECT_ROOT}/.venv/bin/python"
TT_BAG_PATH=/home/chaoszzss/dataset/rosbag/orbvi_4vl_data_cqupt/orbvi_4vl_data.bag
TT_PREPARED_PATH="${TT_PROJECT_ROOT}/prepared"
TT_CALIBRATION_SOURCE="${TT_PROJECT_ROOT}/../board-calib-backup"
TT_CHECK_ONLY=0
TT_REBUILD_TRAJECTORY=0
TT_ROS_PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bag) TT_BAG_PATH="$2"; shift 2 ;;
        --prepared) TT_PREPARED_PATH="$2"; shift 2 ;;
        --calibration-source) TT_CALIBRATION_SOURCE="$2"; shift 2 ;;
        --check) TT_CHECK_ONLY=1; shift ;;
        --rebuild-trajectory) TT_REBUILD_TRAJECTORY=1; shift ;;
        -h|--help)
            echo 'Usage: scripts/prepare_all.sh [--bag BAG] [--prepared DIR] [--calibration-source DIR] [--check] [--rebuild-trajectory]'
            echo 'All original LiDAR frames are mandatory. No frame/seconds limit is supported.'
            exit 0 ;;
        *) echo "Unknown option: $1 (partial preparation is not supported)" >&2; exit 2 ;;
    esac
done
cd "${TT_PROJECT_ROOT}"
[[ -f "${TT_BAG_PATH}" ]] || { echo "Missing bag: ${TT_BAG_PATH}" >&2; exit 1; }
[[ -x "${TT_RUNTIME_PYTHON}" ]] || { echo 'Run scripts/setup_environment.sh first.' >&2; exit 1; }
if [[ ! -f "${TT_PREPARED_PATH}/manifest.json" ]]; then
    [[ "${TT_CHECK_ONLY}" == 0 ]] || { echo 'Missing complete prepared cache.' >&2; exit 1; }
    env PYTHONPATH="${TT_ROS_PYTHONPATH}" /usr/bin/python3 scripts/prepare_bag.py --bag "${TT_BAG_PATH}" --output "${TT_PREPARED_PATH}"
fi

# System Python provides ROS Noetic. Preserve its ROS PYTHONPATH rather than
# replacing it with the runtime environment's incompatible Python 3.12 paths.
env PYTHONPATH="${TT_ROS_PYTHONPATH}" /usr/bin/python3 - "${TT_BAG_PATH}" "${TT_PREPARED_PATH}" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import rosbag
sys.path.insert(0, str(Path.cwd()))
from orbvi_occ.data import load_frames, load_scan, sha256_file
bag_path, prepared = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
manifest = json.loads((prepared/'manifest.json').read_text())
assert manifest.get('complete') is True, 'Cache is not marked complete'
assert bag_path.stat().st_size == manifest['source_bag_bytes'], 'Bag size changed'
assert sha256_file(bag_path) == manifest['source_bag_sha256'], 'Bag SHA256 changed'
with rosbag.Bag(str(bag_path)) as bag:
    counts = {topic:int(info.message_count) for topic,info in bag.get_type_and_topic_info().topics.items()}
assert counts == manifest['bag_counts_all_topics'], 'Full original bag counts changed'
for topic,count in manifest['counts'].items():
    assert counts[topic] == count, 'Incomplete exported topic: '+topic
frames = load_frames(prepared)
assert len(frames) == counts['/orbvi/lidar/custom'], 'Missing LiDAR frames'
points_total = 0
for index,frame in enumerate(frames):
    assert frame['index'] == index, 'Frame gap/duplicate'
    scan = load_scan(prepared,frame)
    points = scan['points']
    assert points.shape == (frame['point_count'],3), 'Incomplete scan'
    for name in ('offset_time_ns','reflectivity','tag','line'):
        assert len(scan[name]) == len(points), 'Point attribute/index mismatch'
    assert int(scan['stamp_ns']) == frame['stamp_ns'], 'Scan reference time mismatch'
    assert int(scan['header_stamp_ns']) == frame['header_stamp_ns'], 'Header time mismatch'
    assert frame['stamp_ns'] == frame['scan_start_ns'] + int(scan['offset_time_ns'].max())
    assert abs(frame['timebase_minus_header_ns']) <= 1000000, 'Cross-clock scan timestamps'
    for camera in frame['images'].values():
        assert camera['stamp_ns'] <= frame['stamp_ns'], 'Future camera leakage'
        assert (prepared/camera['path']).is_file(), 'Missing camera image'
    points_total += len(points)
assert points_total == manifest['lidar_point_count'], 'Lost raw points'
camera_index = json.loads((prepared/'camera_index.json').read_text())
for camera_id,images in camera_index.items():
    topic='/orbvi/raw/camera_'+camera_id+'/image/compressed'
    assert len(images) == counts[topic], 'Missing original camera acquisitions'
    assert all((prepared/image['path']).is_file() for image in images)
with np.load(prepared/'odometry.npz') as odometry:
    assert len(odometry['stamp_ns']) == counts['/orbvi/vio/odometry']
with np.load(prepared/'imu.npz') as imu:
    assert len(imu['stamp_ns']) == counts['/orbvi/imu']
depth = json.loads((prepared/'depth_index.json').read_text())
assert len(depth) == counts['/orbvi/depth/points']
assert all((prepared/frame['path']).is_file() for frame in depth)
print('Complete original bag SHA256/count/cache check passed:', len(frames), 'LiDAR frames,',points_total,'raw points',flush=True)
PY

"${TT_RUNTIME_PYTHON}" - "${TT_CALIBRATION_SOURCE}" "${TT_CHECK_ONLY}" <<'PY'
import json,sys
from pathlib import Path
from orbvi_occ.calibration import copy_device_calibration,export_camera_calibration
from orbvi_occ.data import sha256_file,write_json
root=Path.cwd(); source=Path(sys.argv[1]).resolve(); check_only=int(sys.argv[2])
config_path=root/'configs/calibration.json'
if config_path.exists():
    config=json.loads(config_path.read_text())
    reference=root/'calibration/source/camera_config.yaml'
    assert sha256_file(reference) == config['source_sha256'], 'Archived camera calibration changed'
    if (source/'camera_config.yaml').exists():
        assert sha256_file(source/'camera_config.yaml') == config['source_sha256'], 'Provided camera calibration differs; do not reuse old inference artifacts'
    print('Device camera calibration SHA256 verified; runtime configuration preserved',flush=True)
else:
    assert not check_only, 'Missing camera calibration export'
    if not source.is_dir():
        raise FileNotFoundError('Provide --calibration-source containing the device board backup')
    archive=copy_device_calibration(source,root/'calibration/source')
    write_json(root/'configs/calibration_source.json',archive)
    config=export_camera_calibration(config_path)
    config['primary_source_directory']=str(source)
    config['source_archive']='calibration/source'
    write_json(config_path,config)
PY

if [[ ! -f "${TT_PREPARED_PATH}/trajectory.npz" || "${TT_REBUILD_TRAJECTORY}" == 1 ]]; then
    [[ "${TT_CHECK_ONLY}" == 0 ]] || { echo 'Missing complete trajectory.' >&2; exit 1; }
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "${TT_RUNTIME_PYTHON}" - "${TT_PREPARED_PATH}" <<'PY'
from pathlib import Path
import sys
from orbvi_occ.calibration import build_lidar_trajectory
build_lidar_trajectory(Path(sys.argv[1]))  # Full bag; no frame_limit.
PY
fi

"${TT_RUNTIME_PYTHON}" - "${TT_PREPARED_PATH}" <<'PY'
from pathlib import Path
import csv,json,sys
import numpy as np
from orbvi_occ.data import load_frames
prepared=Path(sys.argv[1]);frames=load_frames(prepared)
report=json.loads((prepared/'trajectory_report.json').read_text())
assert report['complete'] is True and report['frame_count'] == len(frames), 'Incomplete trajectory'
assert report['uses_vio'] and not report['uses_vio_translation'] and not report['uses_vio_scale']
rotation=json.loads((prepared/'vio_rotation_prior_report.json').read_text())
assert rotation['valid_rotation_only'] and not rotation['valid_lidar_camera_extrinsic']
assert rotation['validation_p95_deg'] <= .5, 'Rotation validation failed'
assert rotation['motion_excitation_singular_values'][-1] >= .01, 'Unobservable rotation'
with np.load(prepared/'trajectory.npz') as trajectory:
    poses=trajectory['T_world_lidar'];stamps=trajectory['stamp_ns']
    assert poses.shape == (len(frames),4,4) and np.isfinite(poses).all()
    np.testing.assert_array_equal(stamps,[frame['stamp_ns'] for frame in frames])
    np.testing.assert_allclose(poses[:,3],np.tile([0,0,0,1],(len(frames),1)),atol=1e-10)
    np.testing.assert_allclose(np.einsum('nji,njk->nik',poses[:,:3,:3],poses[:,:3,:3]),
                               np.repeat(np.eye(3)[None],len(frames),axis=0),atol=1e-10)
with np.load(prepared/'vio_rotation_prior.npz') as prior:
    assert (prior['vio_stamp_ns'] <= prior['stamp_ns']).all(), 'Future VIO leakage'
rows=list(csv.DictReader((prepared/'trajectory_quality.csv').open()))
assert len(rows) == len(frames)
for index,row in enumerate(rows):
    assert int(row['index']) == index and row['accepted']=='True', 'Registration failure or omitted frame'
    if index == 0: continue
    expected_rank=int(row['expected_rank'] or 6)
    assert int(row['rank']) == expected_rank, 'Degenerate registration'
    assert float(row['correspondence_fraction']) > .30
    assert float(row['point_plane_rmse_m']) < .20
    assert float(row['step_translation_m']) < .65
    assert float(row['step_rotation_deg']) < np.rad2deg(.15)
print('Full trajectory quality check passed:',len(frames),'frames; point-to-plane P95=',report['point_plane_rmse_m']['p95'],'m',flush=True)
PY
echo 'Full-bag preparation and calibration validation complete.'
