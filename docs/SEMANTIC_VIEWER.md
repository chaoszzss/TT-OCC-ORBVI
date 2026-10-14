# 相机三维语义与全包浏览

查看器读取已提交的真实 NPZ 结果，不启动推理、不改写运行输出。新相机运行采用 `run.json.semantic_registry`；既有 `camera_full` 仍按 `run.json.semantics.class_names` 解释 ADE20K 标签。颜色通过类别名称映射，不能把 ADE20K 数值索引直接当作 nuScenes 索引。

## 启动

在工程根目录运行：

```bash
bash scripts/live_viewer.sh --run runs/camera_semantic_full --port 8767
```

浏览器打开 `http://127.0.0.1:8767/`。运行目录也可以使用绝对路径；查看器可先于推理启动。为了浏览某一历史帧并停止跟随，使用：

```text
http://127.0.0.1:8767/?index=135&follow=0
```

每个 JSONL 完整行是帧的提交边界。存在 NPZ 但没有完整日志行、缺失/半写入的 NPZ，均不作为有效帧显示。浏览器不允许通过 URL 读取任意工作区文件，服务仅绑定 localhost。

## 颜色与类别

默认白背景和真实语义着色。还可切换高度、语义置信度和深色背景。

| 类别 | 颜色来源与意义 |
| --- | --- |
| 车辆 | 官方 TT-Occ car 蓝色 `[0,150,245]` |
| 地面、道路 | 官方 driveable_surface 品红色；语义类别不证明可通行 |
| 墙 | 官方 manmade 淡紫色 |
| 顶棚 | 车库扩展琥珀色；仍映射到 manmade |
| 柱 | 车库扩展青色；仍映射到 manmade |
| 门 | 车库扩展棕色 |
| 行人 | 官方 pedestrian 红色 |
| 语义未知 255 | 独立灰色；几何占据仍保留 |

完整 native 类表在 [taxonomy/garage.json](../taxonomy/garage.json)。图例显示该帧真实体素类别计数。没有已保存置信度的旧运行，置信度模式显示灰色并提示缺失，不生成假分数。新运行的体素 confidence 是候选类最高投票质量占总投票质量的比率；未知体素也可以保留低于准入阈值的候选分数。它不等于已接受标签的准确率，也不是已确认空闲概率，颜色不证明预测正确。

旧 LiDAR 运行标签全部为 255 时会显示灰色语义未知，不会由查看器补出类别。

## 视角与显示筛选

提供官方风格俯视、斜俯视、俯视、低俯角和侧视。官方风格俯视采用近垂直的平行投影和类别原色，不加面光照，默认关闭坐标辅助线，接近 `upstream/assets` 的展示方式；斜俯视采用透视投影和面光照，默认显示坐标辅助线，便于观察立体结构。“坐标辅助线”复选框可独立开关。两者使用完全相同的保存体素、类别、基础 RGB 和通行栅格，投影和光照不会提高预测准确率或缩小体素。拖动旋转、滚轮缩放仍可使用。

新 Garage 相机运行在没有已保存视角时，首次打开默认官方风格俯视并隐藏顶棚，避免上盖遮挡地面和车辆；相邻控件持续显示“显示过滤：顶棚隐藏；保存占据完整”。旧 LiDAR 和旧 ADE 相机运行保持原斜俯视，没有自动顶棚筛选，浏览器已保存的视角与筛选不会被覆盖。

- **完整全高度**：清除类别和 Z 筛选，显示所有保存体素。
- **建筑剖视 · 隐藏顶棚**：只在显示中隐藏名字为 `ceiling` 的类别；页面持续显示隐藏类别与可见数量。
- **图例复选框 / Z 高度**：仅影响三维显示，不修改原始几何和右侧局部通行栅格。

保存视角 JSON 包含 pose、projection（透视／正交）、lighting（面光照／原色）、helper_lines（坐标辅助线）、着色、背景、类别/Z 显示筛选、帧号、时间戳、registry。保存/导入也写入浏览器 localStorage，后续页面可复用。旧视角文件缺失 projection／lighting 时按透视和面光照解释。完整视图和剖切视图是两个不同展示条件，比较结果应使用同一视角、光照与筛选条件。

“保存 PNG”导出当前真实三维画面，同时下载 `.view.json` 标记上述展示条件。仅有 PNG 不能证明未进行筛选，应保留侧文件。完整 PLY 导出所有真实占据体素中心、RGB 和 semantic_id，始终不应用显示筛选；它是体素中心点集，不是相机深度真值或原始点云。

## 同步 RGB、二维语义和 UV 追溯

展开“同步相机与二维语义”可查看本帧保存的语义图及对应 RGB。混合专家运行还提供二维选择器：

| 选择 | 读取的真实缓存 | 含义 |
| --- | --- | --- |
| 最终结果 | `labels` | 本轮真正送入三维关联的二维标签 |
| 官方 OpenSeeD | `official_labels` | 官方 panoptic 组件独立输出 |
| 室内 ADE（Garage 类别） | `b5_labels` | ADE 模型通过名称显式映射后的 Garage 类别；不读取原始 ADE ID 当作 Garage ID |
| 专家来源 | `expert_id` | 标记最终像素来自哪项规则，不是语义类别 |

专家来源使用独立图例：`0` 官方 OpenSeeD（蓝）、`1` 室内 ADE 补充未知（橙）、`2` 室内 ADE 细分柱（绿）、`255` 未知（灰）。图例统计本帧四视图的真实像素数。只切换二维保存结果；三维标签、体素数量、显示模式和局部通行栅格保持原样。旧单模型运行自动隐藏没有保存的专家选项。

新运行通过 `views/{index:06d}.json` 声明每个校正视图，字段包括 `view_id`、`physical_camera_id`、`image_path`、`semantic_image_path`、`semantic_maps_path`、`source_stamp_ns`、`K`、`T_rig_camera` 和 `pixel_domain`。

`rectified_virtual_pinhole` 语义只与同一校正 RGB 对照；“查看原始鱼眼图”另行提供原图，不能将校正 mask 直接叠到鱼眼图。旧运行的原始鱼眼语义保留其旧 pixel domain。页面引用 `source_points/{index:06d}.npz`，其中 source UV、source view ids、camera ids、点标签和证据用于数值追溯；查看器不凭原图或颜色合成新三维语义。

页面校正 RGB 读取 `semantic_images/*.jpg` 的 JPEG95 展示副本，尺寸和校正像素域与模型视图一致。模型实际消费的是 `prepared` 原始图像解码并按标定重映射的数组，未消费展示副本；再次 JPEG 编码会改变部分 RGB 数值。需要精确重现神经输入像素时，应依据 `source_image_path`、保存的标定／K／视图尺寸及执行源码重建，而非以展示 JPEG 逐像素替代实际输入。

## 全包静态图库与 3D 导出

图库读取并验证每一个已提交时间步，清单与选择器覆盖完整运行，不只覆盖 PNG 样本：

```bash
.venv/bin/python scripts/render_semantic_gallery.py \
  --run runs/camera_semantic_full \
  --output reports/camera_semantic_gallery \
  --frames 0,135,600,1174,1800,2347 --pose official \
  --viewer-url http://127.0.0.1:8767 --require-complete
```

`--require-complete` 在创建最终图库前核对连续的全部时间步和 `RUN/verification.json`：独立报告必须通过全包验收，且配置、数据包、神经缓存、`run.json` 与 `frames.jsonl` SHA256 均与当前文件一致。只拥有相同目录名或相同帧数的旧报告不能批准重建后的结果。阶段检查可省略该选项并使用带 `interim` 的独立输出目录；页面和 manifest 会明确显示未完成状态。

输出内容：

```text
reports/camera_semantic_gallery/
  gallery.html                 # 全帧索引 + 实时查看器嵌入 + 静态图
  manifest.json                # 每一帧类别计数、输入来源、总体计数
  view.json                    # 可重复的统一视角
  png/*_allheight.png           # 保留完整高度和全部类别
  png/*_ceiling_cut.png         # 明确标注只在显示中隐藏顶棚
  png/*.view.json              # 来源 NPZ 哈希、pose、筛选和类别
  ply/*_full.ply               # 全高度、全类别、颜色与 semantic_id
  ply/*.metadata.json         # 来源哈希、分辨率和 registry
```

双图使用同一视角、投影、光照与尺度。只移除不可见的内部立方体面用于渲染，不进行体素抽样、几何平滑或模型推理。`--render-all` 为所有时间步导出两套 PNG；`--all-ply` 为所有时间步导出完整 PLY。选择较少 PNG 样本不会缩减全帧清单。`--pose` 可指定 `official`（平行／原色）、`oblique`（透视／光照）、`top`、`low`、`side` 或页面保存的视角 JSON。

HTML 中的三维互动需要上述 live viewer 服务，并与同一运行目录对应；静态 PNG、PLY、清单和视角 JSON 独立保存。`--viewer-url` 可更改 localhost 端口。

完整中文结果报告同样要求当前运行与独立验收的强身份匹配：

```bash
.venv/bin/python scripts/write_camera_semantic_report.py \
  --run runs/camera_semantic_full --require-complete \
  --residency-report diagnostics/residency_probe_three_stage/report.json \
  --output reports/Camera-semantic-results.md
```

## 验证记录

```bash
.venv/bin/python -m pytest tests/test_live_viewer.py tests/test_semantic_viewer.py tests/test_camera_delivery_evidence.py -q
```

旧结果 compatibility、committed boundary、native registry、confidence 缺失和非法值、校正/鱼眼像素域、路径穿越防护、全帧索引及完整 PLY 均有检查。

真实浏览器验收由 `scripts/camera_semantic_browser_qa.py` 检查 WebGL instance buffer 与 NPZ 标签对应、真实颜色、五种视角、显示筛选、view JSON 和源图加载。官方风格与斜俯视只改变投影和光照，原体素几何、基础 RGB、标签与通行图逐数值保持一致。验收证明展示来自保存结果，不证明类别准确率或三维几何精度。
