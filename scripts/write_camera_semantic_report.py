#!/usr/bin/env python3
"""Generate a Chinese report from real committed artifacts; no inference."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'scripts'))
from orbvi_occ.taxonomy import load_registry,class_mapping,class_histogram,UNKNOWN
from camera_delivery_evidence import matching_full_qa,matching_run_identity,matching_residency_validation,run_identity


def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else PROJECT/path


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def rows(path):
    if not Path(path).exists():
        return []
    text=Path(path).read_text();lines=text.splitlines()
    # A concurrent process may be writing the final uncommitted journal line.
    if text and not text.endswith('\n'):
        lines=lines[:-1]
    return [json.loads(x) for x in lines if x.strip()]


def count_labels(run,index,metadata,registry):
    path=run/'occupancy'/f'{index:06d}.npz'
    if not path.exists():
        return None
    with np.load(path,allow_pickle=False) as data:
        labels=np.asarray(data['fused_labels']);count=len(data['fused_indices'])
        if len(labels)!=count:
            raise RuntimeError(f'Actual label/voxel count mismatch: {path}')
        if metadata.get('semantic_registry'):
            histogram=class_histogram(labels,registry)
            mapping='garage class registry'
        else:
            names=metadata.get('semantics',{}).get('class_names')
            if not names:
                return {'count':count,'classes':None,'mapping':'No native class names; numeric-ID guessing forbidden'}
            mapped=np.full(labels.shape,UNKNOWN,np.uint8)
            for native,target in class_mapping(names,registry).items():
                mapped[labels==native]=target
            histogram=class_histogram(mapped,registry)
            mapping='Native class names explicitly mapped to garage registry'
        return {'count':count,'classes':histogram,'mapping':mapping}


def num(value,digits=0):
    return '尚无数据' if value is None else f'{float(value):,.{digits}f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/camera_semantic_full')
    parser.add_argument('--baseline',default='runs/camera_full')
    parser.add_argument('--qa',help='Independent full acceptance report; defaults to RUN/verification.json')
    parser.add_argument('--resources',default='diagnostics/camera_semantic_qa/supervisor_resources.jsonl')
    parser.add_argument('--residency-report',help='Actual GPU residency validation; must match the executed runtime source and cache identity')
    parser.add_argument('--output',default='reports/Camera-semantic-results.md')
    parser.add_argument('--require-complete',action='store_true',help='Require complete recording plus matching independent full QA')
    args=parser.parse_args()
    run,old,out=map(resolve,(args.run,args.baseline,args.output))
    qa_path=resolve(args.qa) if args.qa else run/'verification.json'
    if not out.name.startswith('Camera-semantic') or out.parent!=PROJECT/'reports':
        raise SystemExit('Use a new reports/Camera-semantic* path; existing RESULTS.md stays unchanged')
    meta=read_json(run/'run.json');summary=read_json(run/'summary.json');journal=rows(run/'frames.jsonl')
    if not meta:
        raise SystemExit('Run metadata is missing')
    expected=int(meta['expected_full_recording_frames']);ids=[x['index'] for x in journal]
    prepared=resolve(meta['config']['prepared']);source_rows=rows(prepared/'frames.jsonl');manifest=read_json(prepared/'manifest.json')
    contig=ids==list(range(len(ids)))
    endpoint_match=bool(journal and source_rows and journal[0]['stamp_ns']==source_rows[0]['stamp_ns'] and journal[-1]['stamp_ns']==source_rows[-1]['stamp_ns'])
    bag_match=meta.get('bag_sha256')==manifest.get('source_bag_sha256') and manifest.get('complete') is True
    complete=bool(summary.get('complete') is True and summary.get('acceptance',{}).get('accepted') is True and
        not meta.get('limited_development_run',True) and len(journal)==expected==len(source_rows) and contig and endpoint_match and bag_match)
    qa=read_json(qa_path);qa_matches=qa.get('run_directory')==str(run.resolve())
    qa_full=matching_full_qa(qa,run,meta,expected)
    if args.require_complete and not (complete and qa_full):
        raise SystemExit('Full-success report refused: complete committed recording and matching independent full QA are both required')
    registry=meta.get('semantic_registry') or load_registry();oldmeta=read_json(old/'run.json')
    same_old_bag=oldmeta.get('bag_sha256')==meta.get('bag_sha256')
    samples=[{'index':i,'old':count_labels(old,i,oldmeta,registry) if same_old_bag else None,
              'new':count_labels(run,i,meta,registry) if i in ids else None} for i in (0,135,1174,2347)]
    hist={key:Counter() for key in ('current','fused','source')};near=Counter();far=Counter()
    for row in journal:
        for key in hist:
            hist[key].update(row.get(key+'_semantic_histogram',{}))
        near.update(row.get('source_semantics_within_2m',{}));far.update(row.get('source_semantics_beyond_2m',{}))
    times=np.asarray([r['processing_seconds'] for r in journal],float)
    cache_hits=sum(r.get('inference_cache_hit') is True for r in journal)
    fresh_inference_frames=sum(r.get('inference_cache_hit') is False for r in journal)
    latency=summary.get('latency_seconds') or ({'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'max':float(times.max())} if len(times) else {})
    segments=summary.get('camera_segments')
    if segments is None and journal:
        segments=max(r.get('auxiliary',{}).get('camera_motion',{}).get('segment_id',0) for r in journal)+1
    area_names=sorted({k for r in journal for k in r.get('class_area_m2',{})})
    areas={key:{'mean':float(np.mean([r['class_area_m2'].get(key,0) for r in journal])),
                'max':float(max(r['class_area_m2'].get(key,0) for r in journal))} for key in area_names}
    sem=meta.get('semantics',{});depth=meta.get('camera_depth_provider',{})
    baseline_range_path=PROJECT/'diagnostics/camera_semantic_qa/frame135_baseline_ego_range.json'
    baseline_ranges=read_json(baseline_range_path) if old.resolve()==(PROJECT/'runs/camera_full').resolve() and same_old_bag else {}
    current_135_path=PROJECT/'diagnostics/camera_semantic_qa/full_frame135_consistency.json'
    current_135=read_json(current_135_path)
    geometry_135=read_json(PROJECT/'diagnostics/camera_semantic_qa/full_frame135_geometry_proxies.json')
    actual_sample_135=next((x['new'] for x in samples if x['index']==135),None)
    quality_135=current_135.get('quality',{})
    quality_names={str(x['id']):x['name'] for x in registry['classes']};quality_names['255']='unknown'
    quality_hist_135={quality_names.get(str(k),f'invalid_{k}'):int(v) for k,v in quality_135.get('fused_class_voxels',{}).items()}
    # The first incomplete run was rebuilt after a real boundary correction.
    # A same directory name must not make its old single-frame QA current.
    current_135_match=bool(geometry_135.get('run')==str(run.resolve()) and current_135.get('accepted_for_frame') is True and
        matching_run_identity(geometry_135,run,meta) and matching_run_identity(current_135,run,meta) and
        actual_sample_135 and actual_sample_135['classes']==quality_hist_135 and
        quality_135.get('point_count')==next((r.get('route_input_points') for r in journal if r['index']==135),None))
    sampled_rss=max((r.get('rss_mib',0) for r in journal),default=None)
    resource_path=resolve(args.resources)
    resource_rows=rows(resource_path) if run.resolve()==(PROJECT/'runs/camera_semantic_full').resolve() else []
    current_resource_rows=[r for r in resource_rows if r.get('run_directory')==str(run.resolve()) and
        r.get('run_identity',{}).get('config_sha256')==meta.get('config_sha256')]
    unmatched_resource_rows=len(resource_rows)-len(current_resource_rows)
    resource_hwm={}
    for sample in current_resource_rows:
        for name,process in sample.get('processes',{}).items():
            raw=process.get('VmHWM')
            if raw:
                resource_hwm[name]=max(resource_hwm.get(name,0),int(raw.split()[0]))
    mask_path=meta['config'].get('virtual_views',{}).get('ego_mask_path')
    mask_sha=meta.get('input_provenance_sha256',{}).get(mask_path,'未记录') if mask_path else '未配置'
    selected=sum(r.get('auxiliary',{}).get('camera_maps',0) for r in journal)
    camera_images=sum(manifest.get('counts',{}).get(f'/orbvi/raw/camera_{i}/image/compressed',0) for i in ('2','3','4','5'))
    residency_path=resolve(args.residency_report) if args.residency_report else PROJECT/'diagnostics/residency_probe_three_stage/report.json'
    residency=read_json(residency_path)
    residency_matches=matching_residency_validation(residency,meta)
    qa_text=(f'独立全包验收通过，核查 {expected} 个时间步。' if qa_full else
        f'独立阶段检查通过，核查 {qa.get("statistics",{}).get("verified_frames",0)} 个时间步；全包验收尚未完成。' if qa_matches and qa.get('engineering_consistency_accepted') else
        '尚无与当前运行目录匹配且通过的独立验收报告。')
    state=f'完整处理已完成：{len(journal)}/{expected} 个对齐时间步。' if complete else f'阶段结果：当前只提交 {len(journal)}/{expected} 个对齐时间步，不能称为完整数据包成功。'
    lines=[f'# 相机语义占据{"完整结果" if complete else "阶段报告"}','',''+state+' '+qa_text,'',
        f'本报告来自 `{run}` 的实际提交记录、模型身份、保存的语义 NPZ 与独立验收文件。报告生成器只读取已有真实推理输出；全包完成结论来自当前运行和独立验收。','',
        '## 实际执行记录','','| 项目 | 保存的结果 |','| --- | --- |',
        f'| 提交／预期时间步 | {len(journal)} / {expected} |',
        f'| 选用视图／原包保留 RGB | {selected} / {camera_images} |',
        f'| 原数据包 SHA-256 | `{meta.get("bag_sha256")}` |',
        f'| 配置与源码指纹 | `{meta.get("config_sha256")}` |',
        f'| 推理缓存指纹 | `{meta.get("inference_cache_key")}` |',
        f'| 复用真实冻结推理缓存／本次神经推理时间步 | {cache_hits} / {fresh_inference_frames} |',
        f'| 显式自车掩膜 SHA-256 | `{mask_sha}` |',
        f'| 中位／P95／最大单步耗时 | {num(latency.get("median"),3)} / {num(latency.get("p95"),3)} / {num(latency.get("max"),3)} 秒 |',
        f'| 相机配准片段数 | {num(segments)} |',
        f'| 单帧／融合体素标签记录合计 | {num(sum(hist["current"].values()))} / {num(sum(hist["fused"].values()))} |',
        f'| 来源相机点记录合计 | {num(sum(r.get("route_input_points",0) for r in journal))} |',
        f'| 来源／融合语义未知记录 | {num(hist["source"].get("unknown",0))} / {num(hist["fused"].get("unknown",0))} |',
        f'| 车辆来源点：≤2 m／>2 m | {num(near.get("car",0))} / {num(far.get("car",0))} |','',
        f'完整目标是 {expected} 个对齐时间步和 {expected*4} 个所选视图。约 20 Hz 的全部 {camera_images} 张 RGB 保留在 prepared；未选的中间图像不能称为已经逐图推理。模型及相机位姿估计不输入 LiDAR、设备深度点云或 VIO 位姿。','',
        '点和体素合计跨时间重复计数，不是全局唯一体积或准确率；超过 2 m 只是来源诊断，不等于外部车辆实例标注。','',
        '缓存命中只复用精确输入像素、标定、掩膜、模型和配置对应的真实神经结果；每个时间步的相机配准、占据、时间融合和通行图在本次运行重新构建。缓存中的模型耗时与峰值属于其原始推理，不能冒充本次显存测量；表中的处理耗时来自本次提交日志，含缓存命中与新推理两种情况。','',
        'semantic_images 中的校正 RGB 是 JPEG95 展示副本，具有同一视图尺寸与像素域，但再次压缩会改变 RGB 数值。模型实际输入从 prepared 原始图像解码并按标定重映射生成，未重新读取展示 JPEG 进行推理；精确重现输入应使用原图、保存的视图参数和执行快照。','',
        '## 实际模型与方法','',f'语义提供器：{sem.get("provider","尚未初始化或记录")}。',
        f'几何提供器：{depth.get("provider","尚未记录")}；版本 `{depth.get("revision","尚未记录")}`。','',
        '| 语义专家 | 仓库／版本 | 权重 SHA | 严格加载 |','| --- | --- | --- | --- |']
    for name,identity in (sem.get('experts') or {'semantic':sem}).items():
        revision=identity.get('revision',identity.get('source_commit','未记录'));sha=identity.get('weights_sha256',identity.get('checkpoint_sha256','未记录'))
        lines.append(f'| {name} | {identity.get("repository","未记录")} / `{revision}` | `{sha}` | {identity.get("weights_loaded_strict","未记录")} |')
    if sem.get('experts'):
        lines+=['','默认采用冻结双专家扩展：保留官方 OpenSeeD 已知类别；仅在官方未知处用原生及映射分数都 ≥0.5 的 B5 补充，或用两种分数都 ≥0.65 的 B5 将官方墙细分为柱。整行类别／置信度／top-k 证据来自被选中的真实专家，并保存专家 ID；不混合两个模型的概率。']
    lines+=['','RGB 保留用于检查；明确自车多边形同时约束语义、深度点和射线。掩膜外仍可能有自体边缘或模型误判，不能仅凭掩膜内标签全未知断言所有自体已经剔除。','',
        '几何是零次优化 Gaussian 适配和相机深度 ICP，没有恢复原版 RAFT 场景流与动态对象跟踪。拒绝配准会重置历史并新建局部片段。相机预测射线不能证明实测完整净空，演示足迹也不是实测车辆包络。参见 [双专家方法](../docs/CAMERA_SEMANTIC_HYBRID.md) 和 [几何与语义方法](../docs/CAMERA_SEMANTIC_METHOD.md)。','',
        f'当前足迹配置：长度 {meta["config"].get("traversability",{}).get("footprint_length","未记录")} m、宽度 {meta["config"].get("traversability",{}).get("footprint_width","未记录")} m、安全边距 {meta["config"].get("traversability",{}).get("safety_margin","未记录")} m、净空高度 {meta["config"].get("traversability",{}).get("vehicle_height","未记录")} m。这是演示参数，不是本车实测尺寸。','',
        '## 与旧 camera_full 对照','',
        '旧 ADE20K 类别先按名称映射到相同车库类别；没有直接比较两个类别体系的数字 ID。以下仅统计保存的标签记录，旧车辆记录可能来自自车。它们不是同等外部车辆质量基线，数量变化不能证明准确率提高。','',
        '| 时间步 | 旧车辆标签记录（含自体） | 新车辆标签记录 | 旧柱标签记录 | 新柱标签记录 | 旧／新融合体素总量 |','| --- | ---: | ---: | ---: | ---: | --- |']
    for sample in samples:
        before,after=sample['old'],sample['new']
        def value(item,key):
            return num(item['classes'].get(key,0)) if item and item['classes'] is not None else '尚无可比数据'
        lines.append(f'| {sample["index"]} | {value(before,"car")} | {value(after,"car")} | {value(before,"column")} | {value(after,"column")} | {num(before["count"] if before else None)} / {num(after["count"] if after else None)} |')
    if not same_old_bag:
        lines+=['','旧运行缺失或数据包身份不同，因此不纳入语义对照。']
    if baseline_ranges:
        result=baseline_ranges['result'];old_points=result['old_labels'];old_voxels=result['old_fused']
        point_range=old_points['car_source_camera_range_quantiles_m'];voxel_range=old_voxels['car_voxel_rig_range_quantiles_m']
        lines+=['',f'第 135 帧需要特别区分来源：独立检查发现，旧结果的 {old_points["car_points"]} 个车辆标签来源点全部在 {num(point_range[0],2)}～{num(point_range[-1],2)} m，实图对应采集车辆自身；旧 {old_voxels["car_voxels"]} 个融合车辆体素距 rig 原点约 {num(voxel_range[0],2)}～{num(voxel_range[-1],2)} m。来源点距离以各相机光心为参考，体素距离以 rig 原点为参考。不能将这些自体标签当成外部车辆准确率的分母。',
            f'证据：[旧帧车辆来源／距离归因]({baseline_range_path})。']
    if current_135_match:
        quality=current_135['quality'];car_id=str(next(x['id'] for x in registry['classes'] if x['name']=='car'))
        column_id=str(next(x['id'] for x in registry['classes'] if x['name']=='column'))
        lines+=['',f'当前正式运行第 135 帧保存 {quality["car_points_over_2m"]} 个距离超过 2 m 的车辆标签来源点，2 m 内为 {quality["car_points_under_2m"]}；车辆标签为单帧 {quality["current_class_voxels"].get(car_id,0)}、融合 {quality["fused_class_voxels"].get(car_id,0)}，柱标签为单帧 {quality["current_class_voxels"].get(column_id,0)}、融合 {quality["fused_class_voxels"].get(column_id,0)}。这里展示的是来源变化、几何支持及语义链路恢复；旧自体记录与新车辆标签记录的数字差异不能被称为外部车辆准确率改善，距离超过 2 m 也不等于独立实例真值。',
            f'证据：[当前正式帧独立一致性检查]({current_135_path})。此单帧通过不代表全包验收。']
    lines+=['','## 局部通行与未知','',
        '| 类别 | 每步面积均值 / 最大值（m²） |','| --- | --- |']
    for key,item in areas.items():
        lines.append(f'| {key} | {num(item["mean"],2)} / {num(item["max"],2)} |')
    lines+=['','部分净空候选仍含未知，不能作为实测确认可通行区域；语义地面或道路也不会自动变成空闲。未观测网格保留未知。','',
        '## 资源记录的范围','',
        f'提交日志中采样的父进程运行 RSS 最大值为 {num(sampled_rss,1)} MiB。这个值不包括两模型加载阶段高水位，也不包括独立 OpenSeeD worker；不能称为工程 RAM 峰值。']
    if resource_hwm:
        lines+=['','与当前运行目录和配置指纹匹配的监督样本额外读取 /proc 的各进程 VmHWM：','',
            '| 进程 | OS 高水位（KiB） | 换算 GiB |','| --- | ---: | ---: |']
        for name,kib in resource_hwm.items():
            lines.append(f'| {name} | {num(kib)} | {num(kib/1024**2,2)} |')
        lines+=['',f'来源：[监督资源日志]({resource_path})。这些是各进程分别记录的高水位，不能相加当作同时发生的系统峰值，也不是显存数据。']
    if unmatched_resource_rows:
        lines+=['',f'监督文件另有 {unmatched_resource_rows} 条历史或未记录当前运行身份的资源样本，未计入本次运行高水位；旧拒收进程的数值不能作为新调度的实测峰值。']
    if meta.get('runtime_memory_schedule'):
        lines+=['','本次运行保存的驻留策略：',f'`{meta["runtime_memory_schedule"]}`。',
            '最新调度保留 OpenSeeD → B5 → MapAnything 预测顺序。OpenSeeD worker 进程常驻，注册权重在自身预测结束后移回 CPU；B5 权重在整个语义阶段驻 GPU，包括 OpenSeeD 预测期间，随后移回 CPU；MapAnything 在几何阶段移入 GPU，结束后移回 CPU。实际采用的策略和执行代码以本次 run.json 与 source_snapshot 为准。神经运算仍在 CUDA 上执行，参数 dtype、权重、输入像素和语义门槛保持不变；权重驻留 CPU 不属于 CPU 神经推理或重新训练。']
    if residency_matches:
        array_pairs=[a for f in residency.get('frames',[]) for a in f.get('source_arrays',{}).values()]+[
            a for f in residency.get('frames',[]) for v in f.get('semantic_arrays',[]) for a in v.get('arrays',{}).values()]
        maximum_allocated=max((p.get('parent_peak_allocated_mib',0) for p in residency.get('phases',[])),default=0)
        maximum_reserved=max((p.get('parent_peak_reserved_mib',0) for p in residency.get('phases',[])),default=0)
        lines+=['',f'与当前执行源码及缓存身份匹配的实际 GPU 验证重算时间步 {residency.get("frames_requested")}：全部 {len(array_pairs)} 对来源点／深度／各专家语义数组与旧真实缓存逐值、shape、dtype 一致；移动前后权重 tensor 字节、dtype 和共享别名保持一致。',
            f'该验证主动保留 {num(residency.get("controlled_competing_reservation_mib"),0)} MiB 模拟竞争显存；父进程 CUDA 分配峰值 {num(maximum_allocated,2)} MiB、保留峰值 {num(maximum_reserved,2)} MiB **包含这份竞争保留**，不是单模型或整机总显存。两次四相机推理包含传输的耗时分别为 '+ ' / '.join(num(f.get('runtime_seconds'),3) for f in residency.get('frames',[]))+' 秒。',
            f'证据：[实际阶段驻留及精确数组对照]({residency_path})。这是两个代表时间步和已知竞争规模的验证，不能证明任意其他 GPU 工作量下都不会 OOM，也不能替代最终全包验收。']
        if residency.get('openseed_registered_tensors_cpu_after_each_prediction') is True:
            idle=residency.get('openseed_idle_nvml_mib',{})
            lines+=['',f'同一版本的独立 OpenSeeD worker 审计确认：每次预测后全部注册参数／buffer 位于 CPU；语言缓存的字节、dtype、设备和对象保持一致，RNG 不变。代表空闲阶段 worker 的 NVML 显存为 {" / ".join(num(x,0) for x in idle.values())} MiB，包含 CUDA 上下文和保留的小缓存，不能理解成完全没有 GPU 使用。',
                f'来源：[worker 实际 tensor／缓存审计]({resolve(residency.get("worker_tensor_audit_path","diagnostics/residency_probe_three_stage/worker_audit_report.json"))})。']
    lines+=['',
        '## 验收及解释边界','',qa_text,'',
        f'验收来源：[独立验收 JSON]({qa_path})。实际 [提交日志]({run/"frames.jsonl"})、[配置与模型身份]({run/"run.json"})、[汇总]({run/"summary.json"}) 可复查。','',
        '没有独立三维语义真值时，不报告本车库 OCC mIoU，也不宣称达到官方室外示例或论文榜单准确率。车库顶棚、光照、遮挡及尺度误差仍可能影响几何。完整处理验收、语义数值一致性和真实准确率是不同结论。','',
        'diagnostics/camera_semantic_hybrid 的代表缓存合成复用了最后一次自车边界修正前生成的专家结果，再应用审核掩膜；它不是当前 masked 全包流水线的重新推理或全包验收。最终结论以当前目录和指纹的实际输出为依据。','']
    boundary_report=PROJECT/'diagnostics/camera_semantic_backend/boundary_frame68/report.json'
    if boundary_report.exists():
        lines+=['## 真实边界缺陷与重建','',
            '首个不完整运行在第 68 帧出现网格边界缺陷：合法 float32 扩张点 x=11.999999046325684 小于 12 m，但 float32 相对坐标舍入产生 X=80，超出 80 格轴；Y／Z 在 flatten 前未逐轴检查也存在跨轴混叠风险。此运行保留在 boundary_rejected 目录，不作为成功交付。',
            '修正仅涉及相机体素索引：网格形状与 floor 相对坐标用 float64，整数坐标先逐轴验证再 flatten。合法贡献恢复 X=79；源 XYZ、模型、标定、自车掩膜和旧 LiDAR 模块保持不变。已缓存神经结果可复用，但位姿、占据、时间融合及通行结果重新完整构建。',
            f'实际第 62～68 帧六历史因果回放无越界，独立索引集合和标准邻近来源支持均通过：[边界修复验证]({boundary_report})。这项局部修复验证不替代最终全包验收。','']
    collision_report=PROJECT/'diagnostics/camera_semantic_backend/collision_grid/report.json'
    if collision_report.exists():
        lines+=['### 通行碰撞层的索引统一','',
            '后续真实第 9 帧暴露了第二处耦合错误：地面剔除后的碰撞层仍用旧 float32 raster，出现完整占据层没有的 [71,69,1]。相机路线现在显式注入共享的 camera Gaussian raster，全部非地面来源成员与完整占据使用相同扩张、索引精度及逐轴边界。没有用 collision∩occupied 裁剪掩盖差异；未注入的旧 LiDAR／相机默认路径保持原行为。',
            f'实际第 9、68 帧六历史回放 collision−occupied 均为空，碰撞格与所有有效非地面来源成员的独立索引集合完全一致：[碰撞耦合修复验证]({collision_report})。被拒绝的中间运行单独保存；当前全包仍以最终验收为准。','']
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines))
    details={'run_directory':str(run),'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'complete_processing':complete,'matching_independent_full_qa':qa_full,'processed_frames':len(journal),
        'run_identity':run_identity(run,meta),
        'inference_cache_hits':cache_hits,'fresh_neural_inference_frames':fresh_inference_frames,
        'expected_frames':expected,'journal_contiguous':contig,'source_bag_match':bag_match,
        'sampling_endpoints_match':endpoint_match,'semantic_provider':sem,'depth_provider':depth,
        'samples':samples,'semantic_record_histograms':{k:dict(v) for k,v in hist.items()},
        'source_semantics_within_2m':dict(near),'source_semantics_beyond_2m':dict(far),
        'frame135_baseline_range_attribution':baseline_ranges,
        'frame135_actual_independent_quality':current_135 if current_135_match else None,
        'resource_scope':{'frame_loop_parent_sampled_rss_mib_max':sampled_rss,
            'per_process_os_vmhwm_kib':resource_hwm,'matching_supervisor_samples':len(current_resource_rows),
            'excluded_unmatched_samples':unmatched_resource_rows,'simultaneous_system_peak_measured':False},
        'matching_gpu_residency_validation':residency_matches,
        'gpu_residency_validation_path':str(residency_path),'runtime_memory_schedule':meta.get('runtime_memory_schedule'),
        'local_area_m2':areas,'semantic_miou':None,'original_paper_reproduction':False}
    out.with_suffix('.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
    print(json.dumps({'report':str(out),'complete_processing':complete,'matching_independent_full_qa':qa_full,
                      'processed_frames':len(journal),'expected_frames':expected},ensure_ascii=False))


if __name__=='__main__':
    main()
