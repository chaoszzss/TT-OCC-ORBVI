# 官方 OpenSeeD 相机语义组件

本工程可选的 `orbvi_occ/openseed_provider.py` 使用官方 OpenSeeD Swin-T 权重，不用 SegFormer 冒充 OpenSeeD。它只负责二维语义与实例；后续三维尺度由冻结 MapAnything 和设备相机标定提供。完整 TT-Occ 的动态场景流和原始实例配准仍须分别验证，成功加载 OpenSeeD 不等于完整复现所有模块。

## 真实模型与隔离环境

- 官方权重：`model_state_dict_swint_51.2ap.pt`，465,260,032 bytes，GitHub release asset 105743964。
- 稳定 revision：`release-openseed-asset-105743964`。
- SHA256：`9b6a20c81cf8206845da161cfaf0490563ae79c35e7797058d68913a5f4a61a2`。
- 模型 793 个 state keys 全部严格加载，零 missing、unexpected、shape mismatch。
- 官方源码固定 commit `5dc694ed55e4b71ff57ee3d98ef6ac0dfcaad847`，保存到 `models/openseed-source`。
- Detectron2 使用作者安装说明指定的 MaureenZOU fork，固定 commit `42121d75e10d9f858f3a91b6a39f5722c02868f0`；PanopticAPI 固定 commit `7bb4655548f98f3fedc07bf37e9040a992b054b0`。
- 模型依赖位于 `.openseed-venv`，主 `.venv` 不导入 Detectron2，也不更改其 Transformers 版本。
- PyTorch 2.7 / CUDA 12.8 构建 Deformable Attention 时，仅在源码副本的两个 dispatch 调用将 `value.type()` 改为 `value.scalar_type()`。数值算法与权重未改变，补丁记录为 `models/openseed-source/compatibility-patches.json`。
- Pillow 移除 `Image.LINEAR` 的兼容别名在 worker 内恢复为同值 `BILINEAR`。`upstream/` 保留原样。

初始化与重建：

```bash
.venv/bin/python scripts/setup_openseed.py
```

仅核对/下载资产而不编译时加 `--download-only`。版本与哈希记录在 `models/openseed-swint/provenance.json`、`environment-packages.json` 及各源码目录的 `source-provenance.json`。当前安装脚本面向本机 RTX 4060 的 CUDA architecture 8.9。

## 语义与证据的关系

默认 `output_mode="panoptic"`，调用官方 `panoptic_inference` 原函数：保留 query score gate、mask probability 0.5、overlap gate、stuff merging、segment IDs，输出原函数的真实标签。

wrapper 不修改原函数结果，只保存原算法的 winning query。每像素语义 topK 证据来自**这个 winning query 的类别分数 × sigmoid mask 分数**，不能从另一套密集 `sem_seg` 分布覆盖类别再捏造置信度。被官方拒绝的像素、sky 和无效/自车掩膜区域没有占据语义证据。

额外 `min_confidence` 默认 **0.0**；这不降低官方 query/mask/overlap gate。可设置额外像素门槛，但必须记录该差异。分数是未经校准的模型分数，不是准确率、三维占据概率或空闲概率。

native `sem_seg` 类别与 class-normalized weighted mask mass 保存为独立 `source_labels` / `source_confidence` 诊断。`output_mode="semantic"` 可以显式选择这条不同的诊断路线，不将其称作官方 panoptic。

车库词表固定顺序为：`wall, floor, ceiling, column, car, person, door, tree, obstacle, road, sky`。通过 explicit native registry 对应 0–10，255 为未知。文本提示采用官方单个 template 的流程，并固定 Python/NumPy/Torch seed 0，避免运行间词表模板变化。

## GPU 与调用方式

`OpenSeeDSubprocessProvider` 提供 `metadata()`、`predict_views()`、`last_info`、`close()`，与相机运行器一致。父进程通过真实 NPZ 传递 RGB、valid mask、ego mask 和 rectification mask；输出缓存保留 source query IDs、topK 证据和 panoptic segment JSON。

完整相机路线的最新驻留调度保留 OpenSeeD → B5 → MapAnything 的 CUDA 预测顺序。OpenSeeD worker **进程**持续运行，其注册权重在 `predict_views()` 返回 CPU 数组后移回 CPU 并释放空闲 CUDA 缓存；B5 权重在整个语义阶段驻 GPU，包括 OpenSeeD 预测期间。B5 完成后移回 CPU，再将 MapAnything 权重移入 GPU 进行几何预测，完成后移回 CPU。该驻留操作位于独立运行调度层，保留原 `openseed_provider.py` 的预测数学、参数 dtype、词汇、panoptic 函数及源像素缓存身份。运行保存额外执行模块与启动器的来源哈希；不能拿旧的权重常驻 GPU 实验代表这一版本。

worker 使用 CUDA float32；官方文本编码器直接调用 CUDA。CUDA 不可用时明确失败，不伪造结果或自动替换其他模型。

代表数据的真实预检：

```bash
.venv/bin/python scripts/probe_openseed.py \
  --frames 0,135,1174,2347 \
  --output diagnostics/openseed_panoptic_probe
```

该脚本保存同一校正像素域下 RGB、panoptic 图、显式自车 mask、query confidence、原始 NPZ 和报告，并检查已知标签与 actual winning evidence 类别一致。完成后关闭 worker 释放 GPU。

类计数不能替代图像检查：早期未屏蔽自车的 camera_3 分割中大量 car 像素属于自身引擎盖。对外部车辆、柱边和车库地面应分别检查，不把自车识别当成障碍物预测改进。

官方来源：[OpenSeeD 代码与安装说明](https://github.com/IDEA-Research/OpenSeeD)、[官方 demo](https://github.com/IDEA-Research/OpenSeeD/blob/main/demo/demo_panoseg.py)、[官方模型权重](https://github.com/IDEA-Research/OpenSeeD/releases/tag/openseed)。

## CPU 只读环境检查与可复现安装

已有运行可以执行：

```bash
.venv/bin/python -B scripts/setup_openseed.py --check
```

`--check` 不下载、不安装、不修复、不创建环境、不写模型 provenance 或 Python bytecode，也不实例化网络或加载 tensor checkpoint。它检查固定源码 commit、许可证文件、465 MB 权重哈希、tokenizer revision 与文件、CUDA dispatch 补丁是否仅有声明的两处 API 替换，以及 Torch/torchvision/Transformers/tokenizers/timm 版本。单独子进程只做 CPU imports，确认 Detectron2 和 Deformable Attention 的实际编译符号存在，并记录 `torch.cuda.is_initialized()==False`。因此可在本工程推理进行中检查。

检查结果把“实际推理所需的 imports/assets/binaries 可用”和“全部已安装包的 pip metadata 依赖齐全”分别报告。2026-10-11 的既有环境已实际推理成功，但 `pip check` 仍报告 14 项未解析的附属依赖/版本约束，包括 Hydra 的 antlr、Detectron2 的开发工具以及 iopath/omegaconf metadata 冲突。只读检查如实保留这些结果，当前完整数据运行期间没有安装或修改环境。新的安装步骤使用正常 pip 依赖解析，不再使用 `--no-deps`，并将 iopath 固定为 0.1.9、omegaconf 为 2.3.0，使其同时满足 Detectron2 与 Hydra；IPython、antlr、kornia-rs、lightning-utilities 和 TensorBoard 依赖由 resolver 一起处理。

`venv --system-site-packages` 获得的是 **base Python** 的 packages，不能依赖它自动复用调用者 `.venv` 中的 Torch。安装程序只在隔离子环境创建一个不含执行代码的 `ttocc_parent_runtime.pth`，把主环境 site-packages 放在子环境自身 packages 后面。主环境的 Torch/CUDA runtime 可复用，子环境 Transformers 4.45.2 和 tokenizers 0.20.3 仍优先于主环境的新版本；主环境文件保持原样。临时 empty-venv 的真实独立进程测试已经验证：没有 `.pth` 时 parent-only 模块不可见，桥接后可见，而同名 child 模块仍优先。实验临时目录自动删除，没有安装网络权重或使用 GPU。[Python venv 文档](https://docs.python.org/3.12/library/venv.html)、[site 路径文件说明](https://docs.python.org/3.12/library/site.html)。

主环境必须已具有 Torch `2.7.1+cu126` / torchvision `0.22.1+cu126`；安装程序核对版本并通过 constraint 文件保护它们，同时限制旧 Transformers/tokenizer 版本。所有 pip 命令调用子环境解释器，源扩展的安装也经过同一约束。版本不符合时明确失败，不替换主运行环境。[pip constraints 说明](https://pip.pypa.io/en/stable/user_guide/#constraints-files)。

## 下载源码的许可证记录

| 实际来源 | 固定源码中的许可证 | 本工程记录 |
| --- | --- | --- |
| [IDEA-Research/OpenSeeD](https://github.com/IDEA-Research/OpenSeeD/blob/5dc694ed55e4b71ff57ee3d98ef6ac0dfcaad847/LICENSE) | Apache-2.0 | `models/openseed-source/LICENSE` |
| [MaureenZOU/detectron2-xyz](https://github.com/MaureenZOU/detectron2-xyz/blob/42121d75e10d9f858f3a91b6a39f5722c02868f0/LICENSE) | Apache-2.0 | `models/detectron2-openseed-source/LICENSE` |
| [cocodataset/panopticapi](https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/license.txt) | BSD-2-Clause | `models/panopticapi-source/license.txt` |

这张表记录下载的源码许可证，保留原始版权文件；不是对每份训练数据或全部模型发布条件的扩展声明。CLIP tokenizer 文件固定自 [OpenAI 的 Hugging Face 模型仓库](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268)，只下载 tokenizer/config，没有下载另一份 CLIP 神经网络权重。实际语言模型参数来自已严格加载的 OpenSeeD release checkpoint。

复现审计测试：

```bash
.venv/bin/python -B -m pytest tests/test_openseed_setup.py -q
```
