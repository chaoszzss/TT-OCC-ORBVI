#!/usr/bin/env python3
"""Prepare all original messages, then optionally build a LiDAR-only trajectory."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orbvi_occ.data import prepare_bag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', default='/home/chaoszzss/dataset/rosbag/orbvi_4vl_data_cqupt/orbvi_4vl_data.bag')
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'prepared'))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--trajectory', action='store_true')
    parser.add_argument('--camera-calibration-source', default=None)
    arguments = parser.parse_args()
    manifest = prepare_bag(arguments.bag, arguments.output, arguments.force)
    print(json.dumps({'complete': manifest['complete'], 'counts': manifest['counts'],
                      'lidar_point_count': manifest['lidar_point_count']}, indent=2), flush=True)
    if arguments.camera_calibration_source:
        from orbvi_occ.calibration import export_camera_calibration
        export_camera_calibration(Path(arguments.output).parent / 'configs' / 'calibration.json',
                                  arguments.camera_calibration_source)
    if arguments.trajectory:
        from orbvi_occ.calibration import build_lidar_trajectory
        build_lidar_trajectory(arguments.output)


if __name__ == '__main__':
    main()
