import json
from pathlib import Path
import struct
import unittest

import numpy as np

from orbvi_occ.data import POINT_DTYPE, causal_index, decode_lidar, nearest_index


class DataTests(unittest.TestCase):
    def test_raw_custommsg_round_trip(self):
        points = np.array([(0, 1., 2., -1., 9, 0, 3),
                           (99999999, 3., 4., 5., 17, 8, 1)], dtype=POINT_DTYPE)
        frame = b'livox_frame'
        raw = (struct.pack('<IIII', 23, 1720198192, 800318480, len(frame)) + frame +
               struct.pack('<QIB3sI', 1720198192800318346, 2, 0, b'\x00'*3, 2) + points.tobytes())
        header, cloud = decode_lidar(raw)
        self.assertEqual(header['stamp_ns'], 1720198192800318480)
        np.testing.assert_array_equal(cloud['points'], [[1.,2.,-1.], [3.,4.,5.]])
        np.testing.assert_array_equal(cloud['offset_time_ns'], [0,99999999])

    def test_causal_synchronization_never_reads_future(self):
        stamps = np.array([100, 150, 200], dtype=np.int64)
        self.assertEqual(nearest_index(stamps, 190), 2)
        self.assertEqual(causal_index(stamps, 190), 1)
        self.assertEqual(causal_index(stamps, 200), 2)
        with self.assertRaises(ValueError):
            causal_index(stamps, 99)

    def test_full_preparation_contract(self):
        root = Path(__file__).resolve().parents[1] / 'prepared'
        if not (root / 'manifest.json').exists():
            self.skipTest('Full-bag artifact has not been prepared')
        manifest = json.loads((root / 'manifest.json').read_text())
        frames = [json.loads(line) for line in (root / 'frames.jsonl').read_text().splitlines()]
        self.assertTrue(manifest['complete'])
        self.assertEqual(len(frames), manifest['bag_counts_all_topics']['/orbvi/lidar/custom'])
        self.assertEqual(len(frames), 2348)
        self.assertEqual(sum(frame['point_count'] for frame in frames), 46929024)
        for index, frame in enumerate(frames):
            self.assertEqual(frame['index'], index)
            self.assertGreaterEqual(frame['stamp_ns'], frame['header_stamp_ns'])
            self.assertLessEqual(abs(frame['timebase_minus_header_ns']), 1000000)
            for camera in frame['images'].values():
                self.assertLessEqual(camera['delta_ns'], 0)
                self.assertTrue((root / camera['path']).is_file())
            self.assertTrue((root / frame['lidar_path']).is_file())


if __name__ == '__main__':
    unittest.main()
