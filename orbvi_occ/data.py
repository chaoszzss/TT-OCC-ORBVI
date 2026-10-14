"""Lossless point/image export and timestamp-safe full-bag ORBVI data access.

ROS is needed only during preparation. Inference reads NumPy arrays and JPEGs.
All relative paths in frames.jsonl are relative to the prepared directory.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

LIDAR_TOPIC = '/orbvi/lidar/custom'
ODOMETRY_TOPIC = '/orbvi/vio/odometry'
IMU_TOPIC = '/orbvi/imu'
DEPTH_TOPIC = '/orbvi/depth/points'
CAMERA_IDS = ('3', '2', '4', '5')
POINT_DTYPE = np.dtype([('offset_time', '<u4'), ('x', '<f4'),
                       ('y', '<f4'), ('z', '<f4'), ('reflectivity', 'u1'),
                       ('tag', 'u1'), ('line', 'u1')])


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def parse_header(data):
    seq, seconds, nanoseconds, size = struct.unpack_from('<IIII', data)
    end = 16 + size
    return {'seq': seq, 'stamp_ns': seconds * 1000000000 + nanoseconds,
            'frame_id': data[16:end].decode('utf8')}, end


def decode_lidar(data):
    header, offset = parse_header(data)
    timebase, point_num, lidar_id, reserved, count = struct.unpack_from('<QIB3sI', data, offset)
    offset += 20
    if count != point_num or offset + count * POINT_DTYPE.itemsize != len(data):
        raise ValueError('CustomMsg point count/serialized size mismatch')
    points = np.frombuffer(data, POINT_DTYPE, count=count, offset=offset)
    return header, {'points': np.column_stack([points[k] for k in ('x', 'y', 'z')]),
                    'offset_time_ns': points['offset_time'].copy(),
                    'reflectivity': points['reflectivity'].copy(),
                    'tag': points['tag'].copy(), 'line': points['line'].copy(),
                    'timebase_ns': np.uint64(timebase), 'lidar_id': np.uint8(lidar_id)}


def decode_compressed_image(data):
    header, offset = parse_header(data)
    size, = struct.unpack_from('<I', data, offset)
    offset += 4
    image_format = data[offset:offset + size].decode('utf8')
    offset += size
    size, = struct.unpack_from('<I', data, offset)
    offset += 4
    image = data[offset:offset + size]
    if offset + size != len(data):
        raise ValueError('CompressedImage serialized size mismatch')
    return header, image_format, image


def nearest_index(stamps, stamp):
    stamps = np.asarray(stamps, dtype=np.int64)
    index = int(np.searchsorted(stamps, np.int64(stamp)))
    candidates = [i for i in (index - 1, index) if 0 <= i < len(stamps)]
    if not candidates:
        raise ValueError('Cannot synchronize an empty stream')
    return min(candidates, key=lambda i: abs(int(stamps[i]) - int(stamp)))


def causal_index(stamps, stamp):
    """Last acquisition no later than the scan reference; never future data."""
    index = int(np.searchsorted(np.asarray(stamps, dtype=np.int64), np.int64(stamp), side='right')) - 1
    if index < 0:
        raise ValueError('No causal observation available at the requested time')
    return index


def load_frames(prepared):
    path = Path(prepared) / 'frames.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_scan(prepared, frame):
    with np.load(Path(prepared) / frame['lidar_path'], allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def prepare_bag(bag_path, output, force=False):
    """Export every LiDAR/camera/depth/IMU/VIO message, checking bag index counts.

    Reads raw messages so Livox CustomMsg installation is unnecessary. Header
    acquisition stamps are authoritative; bag receipt stamps are only auditing.
    """
    import rosbag
    bag_path, output = Path(bag_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        stat = bag_path.stat()
        if (manifest['source_bag_bytes'] != stat.st_size or
                manifest['source_bag_mtime_ns'] != stat.st_mtime_ns):
            raise ValueError('Existing preparation belongs to a different bag; use --force')
        frames = load_frames(output)
        if len(frames) != manifest['counts'][LIDAR_TOPIC]:
            raise ValueError('Incomplete prepared index')
        return manifest
    started = time.monotonic()
    topics = [LIDAR_TOPIC, ODOMETRY_TOPIC, IMU_TOPIC, DEPTH_TOPIC]
    topics += ['/orbvi/raw/camera_' + i + '/image/compressed' for i in CAMERA_IDS]
    counts, headers = Counter(), defaultdict(list)
    lidar, cameras, depth = [], {i: [] for i in CAMERA_IDS}, []
    odometry, imu = defaultdict(list), defaultdict(list)
    digests = {topic: hashlib.sha256() for topic in topics}
    for directory in ['lidar', 'depth'] + ['images/camera_' + i for i in CAMERA_IDS]:
        (output / directory).mkdir(parents=True, exist_ok=True)
    with rosbag.Bag(str(bag_path)) as bag:
        info = bag.get_type_and_topic_info().topics
        expected = {topic: int(item.message_count) for topic, item in info.items()}
        definitions = {}
        for topic, raw, received in bag.read_messages(topics=topics, raw=True):
            type_name, serialized, _, _, message_type = raw
            seq_index = counts[topic]
            counts[topic] += 1
            digests[topic].update(serialized)
            header, _ = parse_header(serialized)
            header['bag_stamp_ns'] = received.to_nsec()
            headers[topic].append(header)
            definitions[topic] = {'type': type_name, 'md5': message_type._md5sum}
            if topic == LIDAR_TOPIC:
                _, arrays = decode_lidar(serialized)
                # Livox timebase is the first point. Complete scans are available
                # at their last point, the causal fusion reference timestamp.
                clock_delta = int(arrays['timebase_ns']) - int(header['stamp_ns'])
                if abs(clock_delta) > 1000000:
                    raise ValueError('LiDAR timebase and header epochs disagree by over 1ms')
                scan_end = int(header['stamp_ns']) + int(arrays['offset_time_ns'].max())
                relative = 'lidar/{:06d}.npz'.format(seq_index)
                np.savez(output / relative, **arrays, stamp_ns=np.int64(scan_end),
                         header_stamp_ns=np.int64(header['stamp_ns']))
                frame = dict(header, index=seq_index, lidar_path=relative,
                                  point_count=len(arrays['points']))
                frame['header_stamp_ns'] = header['stamp_ns']
                frame['stamp_ns'] = scan_end
                frame['scan_start_ns'] = header['stamp_ns']
                frame['timebase_minus_header_ns'] = clock_delta
                lidar.append(frame)
            elif topic.endswith('/compressed'):
                camera_id = topic.split('camera_')[1].split('/')[0]
                _, image_format, payload = decode_compressed_image(serialized)
                extension = '.png' if 'png' in image_format.lower() else '.jpg'
                relative = 'images/camera_' + camera_id + '/{:06d}'.format(seq_index) + extension
                (output / relative).write_bytes(payload)
                cameras[camera_id].append(dict(header, path=relative, format=image_format))
            else:
                message = message_type().deserialize(serialized)
                if topic == ODOMETRY_TOPIC:
                    p, q = message.pose.pose.position, message.pose.pose.orientation
                    odometry['position'].append([p.x, p.y, p.z])
                    odometry['orientation_xyzw'].append([q.x, q.y, q.z, q.w])
                    odometry['pose_covariance'].append(message.pose.covariance)
                    odometry['child_frame_id'].append(message.child_frame_id)
                elif topic == IMU_TOPIC:
                    q, a, g = message.orientation, message.linear_acceleration, message.angular_velocity
                    imu['orientation_xyzw'].append([q.x, q.y, q.z, q.w])
                    imu['acceleration'].append([a.x, a.y, a.z])
                    imu['angular_velocity'].append([g.x, g.y, g.z])
                elif topic == DEPTH_TOPIC:
                    names, formats, offsets = [], [], []
                    formats_map = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4',
                                   6: '<u4', 7: '<f4', 8: '<f8'}
                    for field in message.fields:
                        names.append(field.name); offsets.append(field.offset)
                        formats.append(formats_map[field.datatype])
                    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets,
                                      'itemsize': message.point_step})
                    cloud = np.ndarray((message.height, message.width), dtype=dtype,
                                       buffer=message.data,
                                       strides=(message.row_step, message.point_step)).reshape(-1)
                    points = np.column_stack([cloud[k] for k in ('x', 'y', 'z')]).astype(np.float32)
                    arrays = {'points': points, 'stamp_ns': np.int64(header['stamp_ns'])}
                    if 'rgb' in names:
                        arrays['rgb_packed'] = cloud['rgb'].copy()
                    relative = 'depth/{:06d}.npz'.format(seq_index)
                    np.savez(output / relative, **arrays)
                    depth.append(dict(header, path=relative, point_count=len(points)))
            if sum(counts.values()) % 5000 == 0:
                print('Prepared {} messages, {} LiDAR frames'.format(sum(counts.values()), len(lidar)), flush=True)
    if any(counts[topic] != expected[topic] for topic in topics):
        raise ValueError('Export counts do not match the original bag index')
    for topic, records in headers.items():
        stamps = np.array([item['stamp_ns'] for item in records], dtype=np.int64)
        if np.any(np.diff(stamps) <= 0):
            raise ValueError('Non-monotonic acquisition time: ' + topic)
    for topic, arrays, name in [(ODOMETRY_TOPIC, odometry, 'odometry'), (IMU_TOPIC, imu, 'imu')]:
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        arrays['stamp_ns'] = np.array([item['stamp_ns'] for item in headers[topic]], dtype=np.int64)
        arrays['bag_stamp_ns'] = np.array([item['bag_stamp_ns'] for item in headers[topic]], dtype=np.int64)
        arrays['frame_id'] = np.array([item['frame_id'] for item in headers[topic]])
        np.savez(output / (name + '.npz'), **arrays)
    image_stamps = {key: [item['stamp_ns'] for item in values] for key, values in cameras.items()}
    depth_stamps = [item['stamp_ns'] for item in depth]
    odom_stamps = [item['stamp_ns'] for item in headers[ODOMETRY_TOPIC]]
    with (output / 'frames.jsonl').open('w') as index_file:
        for frame in lidar:
            stamp = frame['stamp_ns']
            frame['images'] = {}
            for camera_id in CAMERA_IDS:
                nearest = causal_index(image_stamps[camera_id], stamp)
                image = dict(cameras[camera_id][nearest])
                image['delta_ns'] = image['stamp_ns'] - stamp
                image['valid_sync'] = -100000000 <= image['delta_ns'] <= 0
                frame['images'][camera_id] = image
            nearest = causal_index(depth_stamps, stamp)
            frame['depth'] = dict(depth[nearest], delta_ns=depth[nearest]['stamp_ns'] - stamp)
            nearest = nearest_index(odom_stamps, stamp)
            frame['odometry_index'] = nearest
            frame['odometry_delta_ns'] = odom_stamps[nearest] - stamp
            index_file.write(json.dumps(frame) + '\n')
    write_json(output / 'camera_index.json', cameras)
    write_json(output / 'depth_index.json', depth)
    statistics = {}
    for topic, records in headers.items():
        stamp = np.array([item['stamp_ns'] for item in records], dtype=np.int64)
        receipt = np.array([item['bag_stamp_ns'] for item in records], dtype=np.int64)
        statistics[topic] = {'count': len(records), 'first_stamp_ns': int(stamp[0]),
                             'last_stamp_ns': int(stamp[-1]),
                             'frame_ids': sorted(set(item['frame_id'] for item in records)),
                             'receipt_minus_header_seconds_median': float(np.median((receipt-stamp)/1e9))}
    synchronization = {i: {q: float(np.percentile([abs(frame['images'][i]['delta_ns'])/1e6 for frame in lidar], n))
                            for q, n in [('median_ms', 50), ('p95_ms', 95), ('max_ms', 100)]}
                       for i in CAMERA_IDS}
    stat = bag_path.stat()
    manifest = {'format_version': 1, 'complete': True, 'source_bag': str(bag_path),
                'source_bag_bytes': stat.st_size, 'source_bag_mtime_ns': stat.st_mtime_ns,
                'source_bag_sha256': sha256_file(bag_path), 'bag_counts_all_topics': expected,
                'counts': dict(counts), 'source_types': definitions,
                'timestamp_basis': 'acquisition nanoseconds; LiDAR scan end=header.stamp+max(offset_time); causal camera/depth',
                'lidar_timebase_minus_header_ns_min_max': [min(frame['timebase_minus_header_ns'] for frame in lidar),
                                                           max(frame['timebase_minus_header_ns'] for frame in lidar)],
                'serialized_topic_sha256': {k: v.hexdigest() for k, v in digests.items()},
                'lidar_point_count': sum(item['point_count'] for item in lidar),
                'time_statistics': statistics, 'camera_lidar_sync': synchronization,
                'elapsed_seconds': time.monotonic() - started}
    write_json(manifest_path, manifest)
    return manifest
