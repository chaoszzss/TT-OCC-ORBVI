# 完整记录对比与离线交互页面

使用工程隔离环境执行：

```bash
cd /home/chaoszzss/dataset/rosbag/Occupancy_network/TT-Occ
.venv/bin/python scripts/compare_runs.py
```

默认读取 `runs/lidar_full`、`runs/camera_full` 与 `prepared`，输出：

- `reports/comparison.json`：完整性、逐帧时间分布、运行循环 RSS、记录到的 Torch GPU 分配峰值、过去帧融合体素增量、局部状态面积、相机轨迹段及拒绝记录。
- `reports/comparison.md`：中文完整运行对比报告。
- `reports/gallery.html`：纯本地交互页面，滑块覆盖全部 2,348 个采集时间步，比较实际占据/通行输出及同一步的四张原始相机 JPEG。
- `reports/assets`：从每一帧实际 NPZ 生成的 9,392 张小 PNG，保留真实网格状态。浏览器使用最近邻放大，图像没有模拟或插值补帧。

可通过 `--lidar-run`、`--camera-run`、`--prepared`、`--output` 指定目录。脚本先检查两个 `summary.complete=true` 且 `acceptance.accepted=true`，各有连续 `0..2347` 的帧索引、相同原始 bag SHA256 和相同采集时间戳。任一运行未完成时退出码为 2，拒绝生成完整对比报告。所有实际 NPZ 读取并核对日志后才写入最终 comparison 文件。

图像的数组坐标为 `[x,y]`；页面横轴为局部 x，纵轴为局部 y。完整 Gaussian 占据以竖直列 occupied 体素数着色，保留地面和天花板等全部几何。它与独立的通行状态产品含义不同。四相机图像链接指向 `prepared/images` 中对应时间步的实际因果同步 JPEG，页面不使用外部网站、字体、CDN 或服务，也不需要发布到网站。

直接用浏览器打开 `reports/gallery.html`。保留工程目录相对结构，以便本地原图和地图链接可用。滑块、帧输入框和左右键均可以选择任意时间步；播放模式每一步索引加一，没有时间跳帧。HTML 内嵌全部索引，避免浏览器直接读取本地 JSON 时的跨域限制。

两路线保持各自原生局部坐标。设备 LiDAR—相机外参尚未验收，工具不会将两路线按同一个原点逐点叠加，也不会给出跨传感器逐体素 IoU / mIoU。状态面积用于描述各自输出覆盖情况，不能作为预测准确率。LiDAR 空闲证据来自实测射线；Camera 空闲证据来自预测深度射线。

过去帧增加的体素数用于确认时间融合实际改变了输出覆盖。更多体素同时可能来自深度噪声或位姿误差，因此报告不将增量自动解释为精度提升。

相机深度云 ICP 若被拒绝，报告统计历史重置及独立轨迹段；段之间没有有效配准时，明确不称连续全局地图。即使门控全部通过，其轨迹仍属于预测深度注册结果，没有独立全局位姿真值。峰值 RSS 指运行循环的采样峰值，模型加载阶段不在这个口径中；GPU 指进程记录的 Torch 分配量，不含其他用户进程。未记录的完整运行 GPU 指标保留为空，不用短片段或独立 smoke 值冒充。
