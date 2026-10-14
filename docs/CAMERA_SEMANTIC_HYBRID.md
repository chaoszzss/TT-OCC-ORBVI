# 相机语义占据：OpenSeeD 主分支与车库补充

默认相机语义路线使用真实的官方 OpenSeeD Swin-T 全景分割作为主结果，再用冻结的 SegFormer B5／ADE20K 预训练模型补充车库类别。两个模型都没有在 ORBVI 数据上训练或微调。此方案是双模型语义扩展，不能称为完整复现 TT-Occ 论文。独立 OpenSeeD 和 B5 配置仍可用于对照。

## 固定的像素选择规则

规则在完整数据包运行前确定，不根据三维输出“是否好看”改变标签：

1. 默认保留官方 OpenSeeD 的实际全景类别、置信度和 top-k 证据。
2. 仅当官方类别是语义未知，且 B5 的原生 ADE20K 最大分数和映射后类别分数都至少为 **0.5**，才采用 B5 的真实结果。
3. 仅当官方类别是墙、B5 类别是柱，且 B5 两种分数都至少为 **0.65**，才将墙进一步细分为柱。
4. 其他已知官方类别保持不变。相机无效区域和显式自车掩膜始终为未知，不提供占据点或空闲射线。

每个像素的类别、置信度与整行 top-k 证据均来自被选中的同一个真实专家。不跨模型比较“哪个概率更高”，不混合两个模型的分数。这些分数没有经过准确率校准；高分结果仍是预测，不是人工真值。

## 可独立检查的输出

每个语义 NPZ 保留扁平的 `official_*` 和 `b5_*` 数组，包括两个模型各自的原类别、置信度、原生类别分数、top-k 证据及有效掩膜；全部可用 `allow_pickle=False` 读取。合成结果的 `expert_id` 含义如下：

| ID | 来源 |
| --- | --- |
| 0 | 官方 OpenSeeD 全景结果 |
| 1 | B5 补充官方未知像素 |
| 2 | B5 将官方墙类别细分为柱 |
| 255 | 语义未知 |

三维点仍通过同一来源视图和 UV 获取语义及专家 ID，避免另一路相机接管类别。`source_labels` 的原生类别体系由 `expert_id` 决定，并在 `source_registry_per_expert_json` 中显式说明，不能直接将 ADE20K 数字当作车库类别。原始官方全景实例 ID 保留；B5 补充像素不会被伪造新的实例编号。

运行元数据保存两个模型的仓库、版本、checkpoint SHA、严格加载结果、词表及分数定义。OpenSeeD 在独立依赖环境运行，B5 使用工程已有环境。实际 8 GB 显存占用由真实流水线 pilot 检查；必要时可配置 B5 在 CPU 执行，权重相同，但执行精度、耗时及数值舍入可能不同。

## 代表帧诊断与运行范围

`diagnostics/camera_semantic_hybrid/cached_expert_probe.py` 对第 0、135、1174、2347 时间步的共 16 个真实视图生成合成结果和覆盖图。`cached_probe_summary.json` 及对应目录中的 NPZ、RGB、hybrid 和 expert 图可检查具体来源。

这些诊断复用此前真实推理得到的 OpenSeeD／B5 缓存：**专家缓存在最后一次自车边界修正之前生成，随后由合成脚本应用审核的有效／自车掩膜。它们不是当前完整 masked 流水线的重新推理结果，也不是完整数据包验收。** 原 RGB 没有改变，选择阈值高于此前 B5 缓存的像素阈值；缓存适合检查固定规则和来源，不能替代最终运行的精确配置、掩膜哈希及实际输出。

最终运行另有自己的配置与缓存指纹，目标是覆盖整个数据包的 **2348 个对齐时间步**、共 **9392 个所选相机视图**。原包的约 20 Hz 相机图像仍全部保留在 prepared；未选的中间图像不能被称为已经逐图推理。独立配置是 `configs/camera_semantic_openseed.json` 和 `configs/camera_semantic_b5.json`；默认合成配置是 `configs/camera_semantic_hybrid.json`／`configs/camera_semantic.json`。

几何部分使用零次优化的 Gaussian 适配器；当前没有恢复原版 RAFT 场景流、实例运动关联和动态对象 ICP。相机深度 ICP 被拒绝时，时序历史重置并创建局部片段，不能将未配准片段当作可靠全局地图。相机预测射线不能证明实测完整净空；演示足迹也不是实测车辆包络。没有独立三维标注时，不报告车库 OCC mIoU；类别数量及可视化覆盖增加不能证明准确率提高。

## 旧结果比较时必须区分自体来源

独立检查发现，旧 `camera_full` 第 135 帧的 165 个车辆标签来源点全在
各相机光心约 1.08～1.35 m 范围，实际图像对应采集车辆自身；旧 7 个融合车辆
体素距 rig 原点约 1.12～1.51 m。新正式运行同帧的 185 个车辆标签来源点均
超过 2 m。首次运行的该帧记录为车辆单帧 26／融合 83、柱单帧 924／融合 1433；
该运行后来因另一个时间步的网格边界缺陷而被保留为不完整历史诊断，当前重建后
数量应读取新实际输出及重新生成的独立验收，不能套用旧检查文件。
这些数字说明来源与语义链路改变，**不能把旧 7 到新 83 称为外部车辆准确率改善**，
也不能将旧自体标签当作同等外部车辆基线。超过 2 m 仍只是距离诊断，不是独立实例真值。

实际证据分别为 `diagnostics/camera_semantic_qa/frame135_baseline_ego_range.json`、
`full_frame135_consistency.json` 和 `full_frame135_geometry_proxies.json`。
这只是一个实际提交帧的独立检查，完整包状态必须由最终全序列验收确定。

## 资源统计范围

每帧日志的 `rss_mib` 及旧命名 `peak_process_rss_mib` 只反映帧处理循环采样的
父进程 RSS，不能代表模型加载峰值，也没有包括独立 OpenSeeD worker。
`diagnostics/camera_semantic_qa/supervisor_resources.jsonl` 另读 `/proc` 的
各进程 `VmHWM`：已观测父进程 10,345,572 KiB（约 9.87 GiB）、worker
2,234,916 KiB（约 2.13 GiB）。两个进程的高水位不一定同时发生，不能直接
相加当作系统峰值；它们也不是显存记录。最终报告从实际监督日志重新读取这些值。

## 真实边界缺陷修正

首次不完整运行第 68 帧的 Gaussian 扩张点 x=11.999999046325684
严格小于 12 m，但 float32 的 x+12 舍入为 24，得到越界索引 80。
相机 raster 已改为 float64 计算网格形状和相对坐标 floor，并在 flatten 前
逐轴检查整数边界，恢复合法末格 79，避免 Y／Z 跨轴混叠。
原始相机点、模型、标定、掩膜及旧 LiDAR 模块没有改变。
首次运行保存在 `runs/camera_semantic_full_boundary_rejected`；新的完整运行可
复用真实神经缓存，但重建全部时间步的占据、历史融合及通行输出。
局部实证与独立索引集合检查见
`diagnostics/camera_semantic_backend/boundary_frame68/report.json`，不能替代全包验收。

后续实际第 9 帧又发现通行转换的碰撞层仍用旧 float32 raster，产生完整占据层
没有的 `[71,69,1]` 碰撞格。相机路线现通过 `LocalTraversability` 的显式
`collision_voxelizer` 注入共享相机 raster，地面剔除后的全部非地面来源成员
按同一 Float32 扩张／Float64 索引计算。没有用 collision 与 occupied 取交集
掩盖错误；默认未注入的 LiDAR／旧相机行为保持。
真实第 9、68 帧六历史回放碰撞层均严格属于占据层，且与全部非地面成员的独立
索引集合逐格一致。证据见
`diagnostics/camera_semantic_backend/collision_grid/report.json`。
两次被拒绝的中间运行不是成功交付，新输出需完整重建且通过全包验收。
