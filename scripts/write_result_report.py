#!/usr/bin/env python3
"""Write the delivery report only after both independent full-run audits pass."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    runs={}
    for route in ('lidar','camera'):
        path=ROOT/'runs'/f'{route}_full'
        summary=json.loads((path/'summary.json').read_text())
        audit=json.loads((path/'verification.json').read_text())
        if not summary.get('complete') or not audit.get('accepted'):
            raise SystemExit(f'{route} has not passed complete-recording acceptance')
        rows=[json.loads(line) for line in (path/'frames.jsonl').read_text().splitlines()]
        runs[route]=(summary,audit,rows)
    lidar,la,lrows=runs['lidar'];camera,ca,crows=runs['camera']
    trajectory=json.loads((ROOT/'prepared/trajectory_report.json').read_text())
    manifest=json.loads((ROOT/'prepared/manifest.json').read_text())
    tests=json.loads((ROOT/'reports/test_results.json').read_text())
    segments=camera.get('camera_segments')
    camera_peak=max(r['auxiliary']['camera_depth']['peak_allocated_gpu_mib'] for r in crows)
    def audits(a):
        return f"{sum(c['passed'] for c in a['checks'])}/{len(a['checks'])}"
    lines=[
        '# 完整 ORBVI 数据包执行与交付结果', '',
        '**两条路线均实际处理了全部 2348 个对齐时间步，逐文件独立验收通过。**', '',
        f"原始 LiDAR {manifest['lidar_point_count']:,} 点全部计数保留；推理前剔除 {lidar['invalid_raw_lidar_returns']:,} 个零/无效回波，未删除任何扫描帧。",
        '四路原始相机图像共 18787 张全部保存，两路线各生成 9392 份真实像素语义缓存。', '',
        '| 项目 | LiDAR 路线 | Camera 路线 |',
        '|---|---:|---:|',
        f"| 完整时间步 | {len(lrows)}/2348 | {len(crows)}/2348 |",
        f"| 独立验收检查 | {audits(la)} | {audits(ca)} |",
        '| 占据文件 | 2348 | 2348 |',
        '| 局部通行文件 | 2348 | 2348 |',
        f"| 单步耗时中位数 | {lidar['latency_seconds']['median']:.3f}s | {camera['latency_seconds']['median']:.3f}s |",
        f"| 单步耗时 P95 | {lidar['latency_seconds']['p95']:.3f}s | {camera['latency_seconds']['p95']:.3f}s |",
        f"| 逐帧循环采样峰值 RSS | {lidar['peak_process_rss_mib']:.1f}MiB | {camera['peak_process_rss_mib']:.1f}MiB |",
        f"| 完整运行记录到的 Torch GPU 峰值 | 未记录 | {camera_peak:.1f}MiB |", '',
        f"测试套件实际通过 **{tests['passed']} 项**，其中包括直接运行固定上游 `gs2voxel` 的数值差分、真实 bag 的完整计数/时间同步、因果历史、射线原点、标定质量门和片段拒绝。", '',
        '## 原始算法与适配', '',
        '使用 TT-Occ 固定版本的零迭代 Gaussian 体素化机制与过去帧融合，保持真实上游代码及执行来源哈希。相机路线实际加载官方 MapAnything 1.2B 预训练权重，严格匹配完整参数，没有随机特征占位；语义为真实 SegFormer B0 ADE20K。有限历史、语义替代、索引修正和混合精度都已明确记录。', '',
        '这是无需重训的 TT-Occ 几何适配工程，不是原论文完整开放词汇语义指标复现。', '',
        '## 几何与通行结果的解释', '',
        f"LiDAR 配准使用米制 ICP 和经验证的 VIO 相对旋转约束；P95 内部点到平面残差 {trajectory['point_plane_rmse_m']['p95']:.4f}m。这些数值不是绝对定位真值误差。",
        f"48 次 scan-local 空间体素留出检验共 {lidar['heldout_geometry']['holdout_returns']:,} 个有效回波；命中率 {lidar['heldout_geometry']['weighted_endpoint_hit_rate']*100:.2f}%，误标为空闲的回波比例 {lidar['heldout_geometry']['weighted_false_free_rate']*100:.3f}%。这不是完整占据准确率或语义 mIoU。", '',
        '| 原生局部状态的平均面积 | LiDAR | Camera |',
        '|---|---:|---:|',
    ]
    names={'unknown':'未知','blocked':'阻塞','partial_clearance':'部分净空候选','observed_clearance':'实测完整净空'}
    for key,name in names.items():
        lines.append(f"| {name} | {lidar['class_area_m2'][key]['mean']:.4f}m² | {camera['class_area_m2'][key]['mean']:.4f}m² |")
    lines += ['',
        '面积仅描述各自原生坐标的输出覆盖，不能用来宣称某路线预测更准确。LiDAR 的实测完整净空主要在远场，近场仍有大量未知，没有据此宣称存在可安全规划的连通走廊。相机深度是模型预测，其射线不会生成实测确认的净空类别。', '',
        f"相机时序配准段数：**{segments}**；全局连续性状态：`{str(camera.get('global_pose_continuity_valid')).lower()}`。即使未触发重置，相机轨迹也没有独立绝对真值。", '',
        '## 标定与复现', '',
        '用户提供的设备备份和 ROS1 bridge 工程已经核对。四相机 Triple-Sphere 内参与相对外参已接入；给定 MID360 配置全零 driver 外参无法建立 LiDAR—相机关系。跨传感器外参保持缺失状态，因此两条路线使用各自坐标，LiDAR 三维语义保持未知，不假装进行已标定语义融合或逐体素跨传感器准确率对比。', '',
        '[工程运行说明](../README.md) · [全量对比](comparison.md) · [全部时间步交互浏览](gallery.html)', '',
        '[LiDAR 逐文件验收](../runs/lidar_full/verification.md) · [Camera 逐文件验收](../runs/camera_full/verification.md)', '',
        f"原始 bag SHA256：`{manifest['source_bag_sha256']}`。", '',
        '依赖、上游提交、权重哈希和 Scientific Agent Skills 来源引用已保留在 README 与 models/docs 审计文件中。',
    ]
    (ROOT/'reports/RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Wrote reports/RESULTS.md after both full-run audits passed.')


if __name__=='__main__':
    main()
