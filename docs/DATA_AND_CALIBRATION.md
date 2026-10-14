# ORBVI 完整数据与标定记录

本工程以全部 **2,348 个原始 LiDAR 扫描**为处理索引。四相机、IMU、VIO、设备生成深度按采集时间关联，不通过删帧满足同步条件。全包包含 46,929,024 个原始 LiDAR 点，四路原图共 18,787 张。推理以 LiDAR 的约 10Hz 扫描频率运行，关联 9,392 张图；其余约 20Hz 原图也全部保存在缓存中。

## 输入与用户提供的资料

| 内容 | 实际路径 |
|---|---|
| 原始 ROSbag | `/home/chaoszzss/dataset/rosbag/orbvi_4vl_data_cqupt/orbvi_4vl_data.bag` |
| 用户提供设备标定，主要来源 | `/home/chaoszzss/dataset/rosbag/Occupancy_network/board-calib-backup` |
| 用户提供采集 bridge 工程 | `/home/chaoszzss/orbvi_ros1_ws` |
| 工程内原样标定备份 | `calibration/source/` |
| 先前备份，用于一致性核查 | `/home/chaoszzss/dataset/rosbag/orbvi_4vl_data_cqupt/orbvi/board-calib-backup` |

原 bag 大小 5,233,113,702 bytes，SHA256 为：

```text
9ee5b5b02d2d177d6d4571671235daa3727f4ab5c4785b6efc46fef0f4feba12
```

用户提供的标定目录有 11 个文件，已逐一复制并核对 SHA256。两份 `camera_config.yaml` 内容相同，SHA256 为 `1651d01dfd2d2d949888ecf99334d551b0a5675ef5249d5bea83b80e0eb7f55f`。原件未修改；逐文件记录位于 `configs/calibration_source.json`。

## 两套时间必须区分

bag 的接收时钟与传感器采集时钟不在同一 epoch。下面是原始绝对值，单位 nanoseconds：

| 时间 | 起始 | 结束 |
|---|---:|---:|
| 全 bag 接收时间 | 1785964408876950860 | 1785964643708167238 |
| LiDAR `header.stamp` / 扫描开始 | 1720198192800318480 | 1720198427500026226 |
| LiDAR 扫描结束 / 推理参考时间 | 1720198192900208445 | 1720198427600396191 |
| VIO `header.stamp` | 1720198192659415722 | 1720198427457156658 |
| IMU `header.stamp` | 1720198192814047719 | 1720198427643999768 |
| camera 2/3/4 `header.stamp` | 1720198192709396719 | 1720198427557129768 |
| camera 5 `header.stamp` | 1720198192709396719 | 1720198427507143622 |

bag 接收时间对应 UTC `2026-08-05 21:13:28.876951` 至 `21:17:23.708167`，即上海时区 8 月 6 日凌晨。LiDAR 推理参考时间对应 UTC `2024-07-05 16:49:52.900208` 至 `16:53:47.600396`。传感器时钟日期不同不会自动使相对运动无效，但不能将两套绝对时间混合用于同步。

每个 LiDAR 点的采集时间为扫描开始加 `offset_time_ns`；完整扫描的参考时间为 `header.stamp + max(offset_time_ns)`。全部扫描的 `timebase - header.stamp` 在 −241 到 +242ns 内，因此已确认同一采集时钟。每个参考时刻选择不晚于该时刻的最新相机图和 VIO 姿态。相机时差中位数约 42ms，最大 93.3ms，全部符合 100ms 门限。

## 相机投影与坐标

原图实际尺寸为 **1280×1080**。设备使用 **Triple-Sphere** 模型，九个参数按以下顺序保存：

```text
fx, fy, cx, cy, xi, lambda, alpha, b, c
```

对于相机坐标 `p=(x,y,z)`，投影公式为：

```text
d1 = sqrt(x²+y²+z²)
z2 = z + xi*d1
d2 = sqrt(x²+y²+z2²)
z3 = z2 + lambda*d2
d3 = sqrt(x²+y²+z3²)
den = z3 + alpha/(1-alpha)*d3
u = fx*(x+b*y)/den + cx
v = fy*(c*x+y)/den + cy
```

这九个值不能按 MEI、普通针孔畸变或 `cv2.fisheye` 的参数解释。当地 SDK 的 `fisheye_rig.hpp` 也明确提供 `xi/lambda/alpha/b/c` 字段。

| 原始 topic 相机 | 设备 `camera_config.yaml` 索引 |
|---|---|
| camera 3 | cam0 |
| camera 2 | cam1 |
| camera 4 | cam2 |
| camera 5 | cam3 |

`T_camera0_camera` 将各物理相机坐标转换到物理 cam0 坐标。原 YAML 的平移按毫米提供，导出 JSON 时乘 0.001 转换成米。该对应关系和投影约定有已有四时刻的设备深度 RGB 重投影记录支持；它不证明设备深度的米制精度。

## 未提供可信 LiDAR—相机外参

`MID360s_config.json` 中 `extrinsic_parameter` 的 roll/pitch/yaw/x/y/z 全部为零。这是驱动坐标修正参数，不能解释为 LiDAR 与相机刚好重合。`fisheye_8s/fisheye_stereo_imu_config.yaml` 描述八路虚拟校正相机，且 `estimate_extrinsic: 1` 明确表示 VINS 在线优化前的初值；备份没有物理鱼眼到这些虚拟相机的校正旋转。其 `body_T_cam*` 不能直接套到四路原图上。

在六个不同场景时刻尝试将设备视觉深度和 LiDAR 进行刚性匹配，匹配比例约 0.24–0.31、点到平面 RMSE 约 0.20–0.22m，分时估计平移也不一致。公共 FAST-Calib 文件仅作诊断初始化，未作为设备标定使用。验收失败记录在 `configs/calibration_depth_probe.json`。当前 `T_camera_lidar` 保持 `null`，因此独立相机像素语义不被冒充为已经校准的 LiDAR 三维语义。

本地 bridge HEAD 为 `9bdca7ea700127dee02f3cc601ba8d4c5cd560fa`，日期 2026-07-30，工作区仅 RViz 配置改变。所有 bag topic 的 caller ID 都是 `/orbvi_ros_bridge`。本地源码与采集时间相容，但节点可执行文件在 8 月 17 日重建；bag 没有记录采集时 build commit，因此不声称已经证明二进制完全相同。`configs/calibration_bridge_audit.json` 保存源码 SHA 和 SDK bundle 指纹。

此版本 `MakeLivoxCustomMessage` 逐字段拷贝原始 xyz、点时间与属性，不施加相机外参；`MakeImuMessage` 不变换 IMU 轴；`MakeVioOdometry` 保留设备发布者的 frame metadata。本 bag VIO 的 header 和 child label 均为 `world`，标签本身不足以解释为相机位姿。SDK `getCalibration()` 返回通用 `raw_json`，bridge 用它构造相机/深度标定；所给 SDK 类型、bag 和文件中均未找到 LiDAR—相机变换。通用 JSON 可能承载额外设备字段，但该采集时响应没有保存，不能据此补出外参。

## 米制轨迹与离线旋转校准边界

纯 LiDAR ICP 的初次完整运行出现累计 roll/pitch 漂移。因此最终轨迹使用 LiDAR 的 xyz 和 yaw 估计，同时用经过验证的 VIO 相对旋转约束 roll/pitch，避免将未确认的 VIO 平移或尺度直接转移给 LiDAR。

静态旋转校准先独立处理前 **451 帧**。训练运动的起点为 0、5、…、295，各终点为起点 +10；验证起点为 300、305、…、435，各终点也为起点 +10。因此训练和验证的运动对不重复，但原始帧 300–305 存在短区间重叠，不能称为严格按帧隔离的 holdout。报告中的区间标签描述的是起点范围。旋转激励三项奇异值约为 0.563、0.033、0.023，验证旋转误差中位数 0.107°、P95 0.188°。该拟合只估计 `R_lidar_body`，没有估计平移，也不提供相机外参。IMU proper acceleration 经这个旋转映射后指向 LiDAR 的约 +z；不能将原始 IMU 的 −z 直接当作 LiDAR 的重力轴。

这是**离线的固定传感器旋转校准**，使用了序列前 45 秒，不能宣传为从首帧起严格在线、无需标定的系统。固定旋转校准后，逐帧推理只取采集时间不晚于扫描结束的最新 VIO，deskew 只使用上一帧与当前帧姿态。首个扫描没有过去的运动估计，保持原点云，不读取第二帧反推运动。

最终全部 2,348 帧配准通过门限。内部点到平面 RMSE 中位数 0.0477m、P95 0.0593m、最大 0.0949m；最小匹配比例 0.663。轨迹长度 160.27m，首尾位移 1.019m，z 范围约 0.776m。这些量是内部一致性诊断，既不是独立地面真值，也不等于完成了 SLAM 回环优化。报告与逐帧记录位于 `prepared/trajectory_report.json` 和 `prepared/trajectory_quality.csv`。

初始 `z=0` 是拟合的水平参考，未确认就是道路表面。近场水平面可能来自车辆。局部通行转换需使用真实地面与净空支持；未观测体素必须保持 unknown。原始零点也必须在 deskew 之前过滤，并对 point offset/属性使用同一 mask，避免零点在运动补偿后变成假回波。首个扫描有 988 个零点；完整缓存仍保留所有原始点用于审计。

## 可复现入口

```bash
cd /home/chaoszzss/dataset/rosbag/Occupancy_network/TT-Occ
./scripts/setup_environment.sh --check
./scripts/prepare_all.sh --check
```

`setup_environment.sh` 对已有有效 `.venv` 默认只检查。此次 `.venv` 使用 Python 3.12.14，并继承 `x-anylabeling-cu12` 的系统包；新增依赖仅安装在工程 `.venv`。全新工程通过 `--python /path/to/python3.12` 创建不继承系统包的 venv，安装 torch 2.7.1、torchvision 0.22.1 的 cu126 wheels 和 `requirements-runtime.txt`。脚本不安装到 base，不改变 NVIDIA 驱动。需要主动修改已有工程环境时才使用 `--install`。

```bash
./scripts/setup_environment.sh --python /path/to/python3.12
./scripts/setup_tt_backend.sh             # 冻结语义权重
./scripts/setup_tt_backend.sh --camera    # 需要相机 MapAnything 路线时
./scripts/prepare_all.sh                  # 缺缓存时完整提取与配准
```

原 bag 提取与检查使用 `/usr/bin/python3` 和 ROS Noetic；运行模型与构造轨迹使用工程 Python 3.12。两个环境通过普通 NPZ/JPEG/JSON 交接，不在 Python 3.12 内加载 ROS Noetic 消息类。

`prepare_all.sh` 会重新计算原 bag SHA、对照原 bag 索引核实全部 topic 数量、检查所有扫描的原点数和点属性对应关系、确认全部原相机/IMU/VIO/depth 文件完整，再复用缓存。轨迹也必须覆盖全部索引、姿态有限且正交、每帧配准门限通过、静态旋转验证通过、VIO 时间因果性通过。它不接受 `--limit`。若确需重新构造全部轨迹，可使用 `--rebuild-trajectory`；不要在正式推理任务进行时重建输入产物。

本工程运行时不重新训练网络。ICP、固定旋转估计和射线建图属于几何处理。所下载模型的仓库、不可变 revision 与权重 SHA 记录在 `models/*/provenance.json`；各路线的实际输入、输出以及验收范围见另外两份工程说明。
