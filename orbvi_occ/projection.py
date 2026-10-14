"""Device Triple-Sphere projection and calibrated virtual pinhole views.

The original nine parameters are NOT OpenCV fisheye/MEI distortion parameters.
The chain is taken from the ORBVI calibration/depth implementation; camera
translations in the board backup are converted to metres by calibration.py.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import cv2


def transform_points(points, transform):
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("A finite 4x4 rigid transform is required")
    return p @ t[:3, :3].T + t[:3, 3]


def project_triple_sphere(points_camera, params):
    """Return (uv, valid) for camera optical coordinates in metres."""
    fx, fy, cx, cy, xi, lam, alpha, b, c = np.asarray(params, dtype=float)
    if not (0 < alpha < 1 and fx > 0 and fy > 0):
        raise ValueError("Invalid Triple-Sphere calibration")
    p = np.asarray(points_camera, dtype=np.float64)
    x, y, z = p.T
    d1 = np.linalg.norm(p, axis=1)
    z2 = z + xi * d1
    d2 = np.sqrt(x*x + y*y + z2*z2)
    z3 = z2 + lam * d2
    d3 = np.sqrt(x*x + y*y + z3*z3)
    denom = z3 + alpha / (1-alpha) * d3
    valid = np.isfinite(p).all(1) & (d1 > 1e-9) & (denom > 1e-9)
    safe = np.where(valid, denom, 1.)
    uv = np.stack([fx*(x+b*y)/safe+cx, fy*(c*x+y)/safe+cy], axis=1)
    return uv, valid


class PinholeRectifier:
    """Optical-axis-preserving virtual camera; output intrinsics are explicit."""
    def __init__(self, params, source_size=(1280, 1024), size=(518, 392), fov_deg=100.):
        self.size = tuple(map(int, size))
        self.source_size = tuple(map(int, source_size))
        w, h = self.size
        f = w/(2*np.tan(np.deg2rad(fov_deg)/2))
        self.K = np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
        vv, uu = np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
        rays = np.stack([(uu-self.K[0,2])/f,(vv-self.K[1,2])/f,np.ones_like(uu)],-1)
        uv, valid = project_triple_sphere(rays.reshape(-1,3),params)
        sw, sh = self.source_size
        valid &= (uv[:,0]>=0)&(uv[:,0]<sw-1)&(uv[:,1]>=0)&(uv[:,1]<sh-1)
        self.valid = valid.reshape(h,w)
        uv[~valid] = -1
        self.mapx = uv[:,0].reshape(h,w).astype(np.float32)
        self.mapy = uv[:,1].reshape(h,w).astype(np.float32)

    def rectify(self, image):
        if image.shape[:2] != self.source_size[::-1]:
            raise ValueError(f"Image {image.shape[:2]} disagrees with calibration {self.source_size[::-1]}")
        return cv2.remap(image,self.mapx,self.mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    def lift(self, depth, stride=4, min_depth=.1, max_depth=25.):
        if depth.shape != self.valid.shape:
            raise ValueError("Depth must use rectified image geometry")
        v,u = np.meshgrid(np.arange(0,depth.shape[0],stride),np.arange(0,depth.shape[1],stride),indexing='ij')
        z = depth[v,u]
        ok = self.valid[v,u] & np.isfinite(z) & (z>=min_depth) & (z<=max_depth)
        z,u,v = z[ok],u[ok],v[ok]
        points = np.stack([(u-self.K[0,2])*z/self.K[0,0],(v-self.K[1,2])*z/self.K[1,1],z],-1)
        return points.astype(np.float32), np.stack([u,v],-1)


def load_camera_rig(path):
    cfg = json.loads(Path(path).read_text())
    for cam in cfg['cameras'].values():
        if cam['model'] != 'triple_sphere':
            raise ValueError("This device requires its Triple-Sphere camera model")
    return cfg


def label_lidar_points(points_lidar, camera_cfg, maps, T_camera0_lidar, validated=False):
    """Associate real predictions, gated on cross-sensor calibration quality.

    Unknown labels preserve all LiDAR geometry. Unvalidated transforms never
    silently colour points or decide which historical points are dynamic.
    """
    labels = np.full(len(points_lidar),255,dtype=np.uint8)
    best = np.full(len(points_lidar),np.inf)
    if not validated:
        return labels, {'validated':False,'labelled_points':0}
    p0 = transform_points(points_lidar,T_camera0_lidar)
    for camid, sem in maps.items():
        cam = camera_cfg['cameras'][str(camid)]
        p = transform_points(p0,np.linalg.inv(np.asarray(cam['T_camera0_camera'])))
        uv,ok = project_triple_sphere(p,cam['params'])
        h,w = sem.shape
        ok &= (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        # Prefer the most central optical ray in overlapping fields of view.
        angle = np.arctan2(np.linalg.norm(p[:,:2],axis=1),p[:,2])
        ok &= angle < best
        idx = np.flatnonzero(ok)
        pixel = uv[idx].astype(int)
        labels[idx] = sem[pixel[:,1],pixel[:,0]]
        best[idx] = angle[idx]
    return labels, {'validated':True,'labelled_points':int((labels!=255).sum())}
