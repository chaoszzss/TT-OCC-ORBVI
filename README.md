# ORBVI 4VL 完整数据包 TT-Occ 工程

本工程面向 **整个 `orbvi_4vl_data.bag`**，以全部 **2348 个 LiDAR 扫描结束时刻**作为两条路线共同的时间步。四路相机的 **18787 张原始图像**全部保留；推理使用每个时间步之前最近、同步门限内的四路图像。没有把片段、低帧率片段或空模型当作最终交付。

原始数据路径：`/home/chaoszzss/dataset/rosbag/orbvi_4vl_data_cqupt/orbvi_4vl_data.bag`。主目标为离线的“传感器输入 → 占据构建 → 因果时间融合 → 局部通行转换 → 全量验收”，不是设备反馈控制。

本轮相机语义改进入口为 `scripts/run_camera_semantic.sh`，结果目录为 `runs/camera_semantic_full`；旧 `camera_full` 和雷达结果均保留。新路线使用 110°、518×518 的四个标定虚拟视图、336×336 的冻结 MapAnything 深度、同源像素语义提升以及分离几何/语义证据的时间融合。语义以官方 OpenSeeD 全景预测为基础；冻结 SegFormer B5 仅在固定置信门槛下补充未赋类像素，并将部分墙面细分为柱。两套原始预测及每个像素／三维点的选择来源均保存。具体规则见 [语义组合说明](docs/CAMERA_SEMANTIC_HYBRID.md)。

固定自车图像轮廓屏蔽可见车身输入，原图完整保留。三维查看器提供白色背景、按真实类别着色、官方风格俯视和斜俯视、顶棚显示切换，以及相机原图与对应语义图。官方风格俯视使用平行投影与类别原色，不加面光照；斜俯视保留立体光照。初始顶棚剖视是明确标注的显示过滤，完整顶棚仍保存在占据文件中，可恢复“全高度”。方法差异见 [相机语义方法](docs/CAMERA_SEMANTIC_METHOD.md)。

本轮是否完整成功，以 `runs/camera_semantic_full/verification.json` 的全包验收和当前文件身份匹配为准；运行中的部分结果只用于阶段检查。`boundary_rejected`、`collision_rejected` 以及 `interim` 名称的目录保留问题证据，不能作为成功结果。最终报告和图库生成器均提供 `--require-complete`，同时检查全部 2348 个时间步及独立验收的配置、数据包、神经缓存、元数据和提交日志身份。

```bash
# 全记录相机语义路线；重新运行时另选输出目录
bash scripts/run_camera_semantic.sh --output runs/camera_semantic_retest

# 另开终端浏览运行中的真实结果
bash scripts/live_viewer.sh --run runs/camera_semantic_retest --port 8767
# 浏览器打开 http://127.0.0.1:8767/

# 完整逐帧语义、像素来源、几何证据与通行保守性验收
.venv/bin/python scripts/verify_camera_semantic.py --run runs/camera_semantic_full --output runs/camera_semantic_full/verification.json

# 可选：单独比较两种冻结语义组件；几何、标定、时间步保持同配置
bash scripts/run_camera_semantic.sh --config configs/camera_semantic_openseed.json --output runs/camera_openseed_retest
bash scripts/run_camera_semantic.sh --config configs/camera_semantic_b5.json --output runs/camera_b5_retest
```

白色背景和视角可以接近官方示例，但车库场景包含顶棚、柱与设备自身遮挡，不会变成官方街景。隐藏顶棚只影响显示；未知空间不会被涂成可通行地面。工程仍使用零迭代 Gaussian 适配，没有运行原论文的测试时 3DGS 优化或 RAFT 实例跟踪；有真实多类别语义输出也不等于原论文指标复现。

## 已准备的两条路线

| 路线 | 几何输入与尺度 | 时间融合 | 语义 |
|---|---|---|---|
| LiDAR | 实测 Livox XYZ；米制 LiDAR ICP，带经验证的 VIO **旋转**约束；不使用 VIO 平移/尺度 | 过去 1 秒的 TT-Occ Gaussian 来源回波，逐扫描运动补偿 | 四相机真实预训练 SegFormer；跨传感器外参缺失时，三维语义保持 `255=unknown_semantic` |
| Camera | 四路 Triple-Sphere 原图经标定重投影；真实官方 MapAnything 米制权重；固定相机 rig 内外参 | 相机预测点云自身 ICP；条件数/残差门控，不输入 LiDAR 深度或 LiDAR/VIO 运行轨迹；失败时切换局部段 | 同一真实预训练 SegFormer，依据设备四相机标定关联到预测点云 |
| Camera semantic（本轮默认改进入口） | 四路方形 110° 标定视图；冻结 MapAnything 336，保留源 UV／光心／深度 | 类别证据分离、当前观测优先、0.6 秒衰减历史和按真实光心的射线一致性检查 | 原 OpenSeeD 全景语义 + 冻结 ADE20K 的显式补充；同源像素提升，无跨相机抢标签 |

`upstream/` 保留 TT-Occ 官方仓库固定提交。工程提取并移植其零迭代 LiDAR Gaussian 初始化和 `gs2voxel`，在 CPU 上执行；数值差分测试直接执行上游原函数。它是明确记录差异的 **TT-Occ 几何适配版**，不是原论文完整语义基准复现。语义模型替代、有限历史、负坐标索引修正、相机模型精度适配等见 [UPSTREAM_ADAPTATION.md](docs/UPSTREAM_ADAPTATION.md)。无需重新训练或微调网络。

## 直接使用

当前 `.venv`、完整数据缓存和真实模型权重均位于本工程目录。[原几何结果报告](reports/RESULTS.md) 和 [原两路线对比](reports/comparison.md) 对应此前的 `lidar_full`／`camera_full`，不是本轮新相机语义结果。

本轮重启机器后的使用流程无需重新下载权重或提取数据。执行检查后，直接启动新相机入口或浏览已保存的新结果：

```bash
cd /home/chaoszzss/dataset/rosbag/Occupancy_network/TT-Occ
.venv/bin/python -B scripts/setup_camera_semantic_models.py --check
.venv/bin/python -B scripts/setup_openseed.py --check
bash scripts/live_viewer.sh --run runs/camera_semantic_full --port 8767
# 浏览器打开 http://127.0.0.1:8767/?index=135&follow=0
# 需要重新推理时使用新的目录：
bash scripts/run_camera_semantic.sh --output runs/camera_semantic_retest
```

两项模型检查不启动网络推理或 CUDA tensor，也不修改环境。查看器只读取结果，另起一轮推理才会加载冻结模型。

本机 8 GB GPU 上，默认路线按 OpenSeeD → B5 → MapAnything 顺序预测，并按语义／几何阶段安排 GPU 权重驻留。OpenSeeD 的隔离 **进程**仍常驻，其注册权重只在官方语义预测时移入 GPU，返回 CPU 数组后移回 CPU。B5 权重在整个语义阶段驻 GPU，包括 OpenSeeD 预测期间；B5 完成后移回 CPU，再将 MapAnything 权重移入 GPU 计算深度，完成后移回 CPU。模型、dtype、输入像素、语义门槛和 CUDA 神经运算保持不变，独立驻留调度的执行代码及来源哈希进入本次运行快照。

最新三阶段实际验证记录在 `diagnostics/residency_probe_three_stage/report.json`：第 1199／1228 时间步的 400 对来源与专家语义数组均与旧真实缓存精确一致，三个模型的 tensor 字节、dtype 和共享别名保持一致。OpenSeeD 注册参数／buffer 预测后均驻 CPU，语言缓存及 RNG 未变；代表空闲 worker 的 NVML 显存仍为 200／278 MiB，含上下文与小缓存。带显式 1055 MiB 竞争保留的两次四相机重算耗时为 6.129／4.933 秒，期间采样的全卡最小空闲显存为 1295.125 MiB；这是代表实验，不是全包显存上界或传感器实时性能承诺。

`diagnostics/residency_probe/report.json` 保留此前“两模型分阶段、OpenSeeD 权重仍驻 GPU”的第 135／550 时间步实验，不能作为最新三模型调度的显存或速度实测。该中间运行后来在第 1229 时间步 OOM，未完成全包。最终报告只纳入执行源码及缓存身份匹配的驻留实验，并区分父进程显存、worker 显存、全卡空闲量和外部负载；代表实验不保证任意外部 GPU 负载都不会 OOM。

中断时可对同一输出目录使用 `--resume`。精确配置的冻结神经缓存可复用，未提交时间步重新处理；相机位姿、融合与通行历史会从因果观测恢复，不会把缺失帧或失败结果标为成功。重新配置时使用新输出目录。

```bash
cd /home/chaoszzss/dataset/rosbag/Occupancy_network/TT-Occ

# 核验环境和完整缓存，不重新安装或提取
bash scripts/setup_environment.sh --check
bash scripts/prepare_all.sh --check

# 运行新一轮完整测试，分别保存，避免覆盖已交付结果
bash scripts/run_lidar.sh --output runs/lidar_retest
bash scripts/run_camera.sh --output runs/camera_retest

# 独立检查每个已交付时间步与输出文件
.venv/bin/python scripts/verify_run.py --run runs/lidar_full
.venv/bin/python scripts/verify_run.py --run runs/camera_full

# 生成完整结果对比与所有时间步的离线交互浏览器
.venv/bin/python scripts/compare_runs.py

# 导出真实占据体素中心，使用 CloudCompare 或 Open3D 检查三维几何
.venv/bin/python scripts/export_ply.py --route lidar --frame 1174
.venv/bin/python scripts/export_ply.py --route camera --frame 1174

# 本地浏览；不上传或发布数据
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
# 浏览器打开 http://127.0.0.1:8765/reports/gallery.html
```

`run_lidar.sh` 与 `run_camera.sh` 默认使用各自配置，也可以传 `--config`；开发用 `--limit` 会明确产生不完整结果，不能通过完整数据验收。中断后可使用相同代码、标定、模型及配置的 `--resume`，具体检查点边界见 [VALIDATION_PROTOCOL.md](docs/VALIDATION_PROTOCOL.md)。已完成的默认目录应先验收再浏览；重跑请使用新的 `--output`。

已交付完整结果包含当时实际执行的 `executed_source/` 快照和哈希。全量结束后的唯一公共入口修正是恢复时清理未提交的影子评价日志，并已用真实扫描中断场景测试；它不改变正常全量推理的几何或神经模型计算。新一轮测试使用当前代码，已交付结果按对应执行快照验收。

新环境部署：

```bash
bash scripts/setup_environment.sh --python /path/to/python3.12
bash scripts/setup_tt_backend.sh --camera
.venv/bin/python scripts/setup_camera_semantic_models.py --download
.venv/bin/python scripts/setup_openseed.py
bash scripts/prepare_all.sh
```

数据提取使用系统 ROS1 Python，模型及几何执行使用工程 `.venv`。安装脚本只修改工程环境；当前环境借用本机既有 Torch 二进制，新环境使用干净 venv。PyTorch 2.7.1 / CUDA 12.6 已在本机实际验证。本轮默认的 OpenSeeD 双专家路线要求 CUDA，不能在 GPU 不足时无声切到 CPU；旧几何与单 SegFormer 路线的 CPU 支持不等于默认双专家支持。相机模型下载与源码版本有固定哈希，不需要额外下载 DINO 初始化权重。

## 推理期间实时可视化

打开两个终端，均进入本工程目录。终端一运行算法：

```bash
bash scripts/run_lidar.sh --output runs/lidar_retest
```

终端二运行独立、只读的查看器：

```bash
bash scripts/live_viewer.sh --run runs/lidar_retest
```

浏览器打开 **http://127.0.0.1:8766/**。查看器也可以先启动，输出目录尚未创建时会显示等待。默认每 500ms 查询最新**已提交**帧，三维窗口显示真实占据体素立方体，右侧显示局部通行分类；刷新频率不代表算法达到了传感器采集频率。

- 鼠标拖动旋转三维视角，滚轮缩放；可按高度裁剪查看地面、障碍或天花板。
- 切换“当前帧/融合占据”，直接比较过去帧融合产生的几何变化。
- 暂停自动跟随后，可以拖动滑块查看任何已完成的历史帧，再恢复跟随。
- 未知、阻塞、部分净空候选、实测完整净空保持独立；相机预测不会变成实测确认净空。

查看器支持 LiDAR 和 Camera，也支持已完成的全量结果，例如 `--run runs/camera_full`。端口被占用时使用 `--port 8767`。三维渲染需要浏览器支持 WebGL2；不可用时页面会明确提示并保留二维图。查看器不加载模型、不改变推理配置、不修改输出数据；停止查看器不影响另一终端的推理进程。

## 输出与坐标

```
prepared/                    全部原始传感器缓存、逐扫描因果索引、轨迹与来源哈希
calibration/source/          用户提供的设备标定原件副本，逐文件 SHA 一致
configs/                    路线参数、标定、来源与采集 bridge 审计
models/                     实际预训练权重、官方源码与版本审计
runs/lidar_full/             完整雷达路线
runs/camera_full/            完整相机路线
  occupancy/000000.npz       单帧与融合占据索引、语义、体素范围、当前位姿
  traversability/000000.npz  原始 TT 占据、碰撞图、射线自由证据、地面和局部分类
  semantic_maps/             每个时间步四路原图的真实像素语义
  frames.jsonl              逐帧输入计数、时刻、融合增益、时延与质量状态
  summary.json              完整处理验收与统计，不是绘图元数据
  verification.json         独立逐文件验收
reports/                    中文结果、对比与离线交互浏览
tests/                      上游数值差分、几何、因果、真实数据和验收测试
```

新相机语义运行另有 `source_points/`（真实源 UV、XYZ、相机内外参、深度、选中专家），`views/`（每帧四视图与原图时间戳），`semantic_images/`（同一校正像素域的 JPEG95 展示副本）以及包含两套原预测的 `semantic_maps/`。模型实际输入由 `prepared` 原始压缩图像解码、标定重映射后生成，未重新读取 JPEG95 展示副本进行推理；展示副本与实际输入的 RGB 数值会因再次压缩有所区别。逐像素重现应使用原始图像、保存的标定／视图参数及执行源码。`source_snapshot/` 保存实际执行源码、标定、类别及模型来源哈希；`frames.jsonl` 的完整换行是提交边界。中断后的 `--resume` 只复用精确配置的神经推理缓存，并重放过去观测以恢复因果历史，不跳过尚未提交的时间步。

局部栅格采用 `0.3m` 体素、`[-12,-12,-2]` 至 `[12,12,4]` 范围，数组轴为 `[x,y,z]`。LiDAR 输出位于当前 LiDAR 坐标；Camera 输出位于 camera_3 光心建立的前/左/上 rig 坐标。两者没有被假装转换成同一个已标定车体原点。

占据文件中，**存在的体素就是几何占据**；`255` 只表示语义未知。缺失占据体素不能直接理解成自由空间。相机原始 9 参数采用 **Triple-Sphere**，不是 OpenCV fisheye 或 MEI，虚拟图像的 K 随缩放同步变换。旧 `camera_full` 使用 518×392 → 224×224；新 `camera_semantic_full` 使用 518×518 → 336×336，并保存生成每个三维点的虚拟相机、像素坐标、内外参、预测深度以及对应的语义证据。

## 局部通行转换的含义

| 分类值 | 含义 |
|---|---|
| 0 | 未知或证据不足 |
| 1 | 障碍按 footprint 膨胀后阻塞 |
| 2 | 地面/部分净空支持的候选，仍含未知 |
| 3 | 实测射线覆盖整个所需体积与 footprint 的净空 |

相机模型深度不等于实测空闲，其输出不会生成类别 3。Gaussians 对地面的上方扩展不能变成虚假障碍：原始 TT 占据仍保留，碰撞图则按照可靠地面来源回波重新构建，并单独保存来源统计。

当前默认 footprint 为 0.7×0.55m、车体高度 1m、安全裕量 0.1m，是可配置的演示尺寸；实际设备/车辆包络未由用户提供。MID360 近场下方视野有限，因此近场可能保留大量未知，类别 3 可能仅在远场出现。请看真实面积、地面支持和连通性，不把黄色候选当作已确认可通行走廊。

## 标定与评价边界

用户提供了 `../board-calib-backup` 和 `/home/chaoszzss/orbvi_ros1_ws`。四物理相机标定已接入；MID360 配置的全零 driver extrinsic 不是 LiDAR—相机外参，八虚拟相机的 VINS 初始外参也不能直接替代四物理相机外参。公共示例与粗配准没有被冒充设备标定。详细记录见 [DATA_AND_CALIBRATION.md](docs/DATA_AND_CALIBRATION.md)。

雷达配准的旋转约束来自前 451 帧离线运动校准，不能宣称从首帧开始在线完成设备标定；之后处理只消费当前/过去采集。内部配准残差不是绝对定位误差。Camera 的时序配准完全来自预测相机几何；若出现分段，全局预览明确标为未配准段，不表示连续可靠世界地图。

完整验收检查所有 2348 帧及 46929024 原始点的计数、首末时间戳、索引与 NPZ 文件。零/无效回波在运动补偿前过滤，但原始计数和数据仍完整保留。主线使用全部有效回波；单帧影子评价另按整空间体素留出 10% 测量，避免同一源点既构建又评价。没有语义 GT，不报告自建数据语义 mIoU；缺少可靠跨传感器外参时，也不报告两路线逐体素准确率排名。

## 来源

- [TT-Occ 官方代码](https://github.com/Xian-Bei/TT-Occ)，固定提交 `0cdb8c9e8abc8124fd956c1e0cb582d3ad18b333`。
- [Test-Time 3D Occupancy Prediction](https://arxiv.org/abs/2503.08485)。
- [官方 MapAnything Apache 权重](https://huggingface.co/facebook/map-anything-apache)，实际权重与源码哈希见 `models/`。
- [SegFormer B0 ADE20K 权重](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)，固定版本与哈希见 `models/segformer-b0-ade20k/provenance.json`。
- [OpenSeeD 官方权重](https://github.com/IDEA-Research/OpenSeeD/releases/tag/openseed) 和 [SegFormer B5 ADE20K 权重](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640)，为本轮实际使用的两个冻结语义组件，严格加载、固定源码和哈希见 [OpenSeeD 接入](docs/OPENSEED_PROVIDER.md) 与 [双专家规则](docs/CAMERA_SEMANTIC_HYBRID.md)。
- 资源与预训练组件工作流使用 Scientific Agent Skills 的 get-available-resources / transformers / scientific-visualization 指南；引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。各上游代码和权重保留自身许可。
