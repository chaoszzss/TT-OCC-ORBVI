#!/usr/bin/env python3
"""Actual browser/GPU-buffer QA for camera semantic viewer, without inference."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import threading
from urllib.request import urlopen

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parents[1]


def api(url, frame, mode='fused'):
    with urlopen(f'{url}/api/frame?index={frame}&mode={mode}', timeout=15) as handle:
        return json.load(handle)


def load_semantic_image_arrays(run, index, source):
    path = run / source.get('semantic_path', f"semantic_maps/{index:06d}_{source.get('view_id', source['camera'])}.npz")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in ('labels', 'official_labels', 'b5_labels', 'expert_id')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT / 'runs/camera_semantic_full')
    parser.add_argument('--frame', type=int, default=135)
    parser.add_argument('--output', type=Path, default=PROJECT / 'diagnostics/camera_semantic_qa/browser')
    args = parser.parse_args()
    args.run = args.run.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('camera_qa_viewer', PROJECT / 'scripts/live_viewer.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(PROJECT / '.viewer-test-browsers'))
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'run': str(args.run),
              'scope': 'Actual source NPZ, localhost API, Chromium WebGL2 native instance buffers and screenshots; no model or semantic accuracy test',
              'checks': [], 'javascript_errors': [], 'console_errors': []}
    run_bytes = (args.run / 'run.json').read_bytes()
    journal_bytes = (args.run / 'frames.jsonl').read_bytes()
    run_metadata = json.loads(run_bytes)
    report['run_identity'] = {'config_sha256': run_metadata.get('config_sha256'),
        'bag_sha256': run_metadata.get('bag_sha256'), 'inference_cache_key': run_metadata.get('inference_cache_key'),
        'run_json_sha256': hashlib.sha256(run_bytes).hexdigest(),
        'frames_jsonl_sha256': hashlib.sha256(journal_bytes).hexdigest()}
    report['frame'] = args.frame
    with ExitStack() as stack:
        server = module.make_server(args.run, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        stack.callback(server.server_close); stack.callback(server.shutdown)
        url = f'http://127.0.0.1:{server.server_port}'
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=['--enable-webgl', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader'])
            page = browser.new_page(viewport={'width': 1400, 'height': 1000}, device_scale_factor=1)
            page.add_init_script("""(() => {
              window.__qaBuffers = [];
              window.__qaDraw = null;
              const buffer = WebGL2RenderingContext.prototype.bufferData;
              WebGL2RenderingContext.prototype.bufferData = function(target, data, usage) {
                buffer.call(this, target, data, usage);
                if (data instanceof Float32Array && data.length % 7 === 0 && data.length > 7) {
                  window.__qaBuffers.push(Array.from(data));
                  window.__qaBuffers = window.__qaBuffers.slice(-3);
                }
              };
              const draw = WebGL2RenderingContext.prototype.drawArraysInstanced;
              WebGL2RenderingContext.prototype.drawArraysInstanced = function(mode, first, count, instances) {
                draw.call(this, mode, first, count, instances);
                window.__qaDraw = {mode, first, count, instances, error: this.getError()};
              };
            })()""")
            page.on('pageerror', lambda error: report['javascript_errors'].append(str(error)))
            page.on('console', lambda message: report['console_errors'].append(message.text) if message.type == 'error' else None)
            payload = api(url, args.frame)
            assert payload['status'] == 'ready' and payload['route'] == 'camera'
            with np.load(args.run / 'occupancy' / f'{args.frame:06d}.npz', allow_pickle=False) as artifact:
                expected_centers = artifact['bounds'][0] + (artifact['fused_indices'] + .5) * float(artifact['voxel_size'])
                expected_labels = artifact['fused_labels']
            np.testing.assert_allclose(payload['centers'], expected_centers, atol=1e-8, rtol=0)
            np.testing.assert_array_equal(payload['labels'], expected_labels)
            page.goto(f'{url}/?follow=0&index={args.frame}', wait_until='domcontentloaded')
            page.wait_for_function('''expect => { const s=window.LiveViewerDebug?.snapshot(), d=window.__qaDraw;
              return s && s.index === expect.frame && s.cubeCount === expect.count && s.webgl2 &&
                d && d.instances === expect.count && d.count === 36 && d.mode === 4 && d.error === 0;
            }''', arg={'frame': args.frame, 'count': len(expected_centers)}, timeout=30000)
            ceiling_id = next((item['id'] for item in payload['semantic_registry']['classes'] if item['name'] == 'ceiling'), None)
            if payload.get('pipeline') == 'camera_semantic_v1' or payload['semantic_registry'].get('schema_version') == 'orbvi-garage-v1':
                initial = page.evaluate('window.LiveViewerDebug.snapshot()')
                assert ceiling_id in initial['filter']['hidden_semantic_ids']
                assert initial['cubeCount'] == len(expected_centers)
                assert initial['visibleCount'] == len(expected_centers) - int(np.count_nonzero(expected_labels == ceiling_id))
                assert page.locator('#ceilingBadge').is_visible() and page.locator('#filterBanner').is_visible()
                report['checks'].append({'name': 'default_garage_ceiling_cut_explicit_and_geometry_complete',
                                         'ceiling_class': ceiling_id, 'stored_cubes': initial['cubeCount'], 'visible_cubes': initial['visibleCount']})
            page.locator('#colorMode').select_option('semantic')
            page.locator('#theme').select_option('white')
            page.locator('#allHeightBtn').click()
            palette = {int(item['id']): item['color'] for item in payload['semantic_registry']['classes']}
            gpu_result = page.evaluate('''({centers,labels,palette}) => {
              const data=window.__qaBuffers.findLast(a=>a.length===centers.length*7);
              if (!data) throw new Error('Native WebGL instance upload missing');
              let maxCenterError=0,maxColorError=0,invalidVisibility=0;
              for(let i=0;i<centers.length;i++){
                for(let j=0;j<3;j++){
                  maxCenterError=Math.max(maxCenterError,Math.abs(data[i*7+j]-centers[i][j]));
                  maxColorError=Math.max(maxColorError,Math.abs(data[i*7+3+j]-palette[labels[i]][j]/255));
                }
                if(data[i*7+6]!==1)invalidVisibility++;
              }
              return {maxCenterError,maxColorError,invalidVisibility,nativeInstances:centers.length};
            }''', {'centers': expected_centers.tolist(), 'labels': expected_labels.tolist(), 'palette': {str(k): v for k, v in palette.items()}})
            assert gpu_result['maxCenterError'] < 1e-5 and gpu_result['maxColorError'] < 1e-6 and gpu_result['invalidVisibility'] == 0, gpu_result
            report['checks'].append({'name': 'all_actual_gpu_centers_and_semantic_colors_match_npz_registry', **gpu_result})
            for class_id, color in palette.items():
                assert page.evaluate('(id)=>window.LiveViewerDebug.semanticColor(id)', class_id) == color
            counts = {str(int(a)): int(b) for a, b in zip(*np.unique(expected_labels, return_counts=True))}
            assert page.evaluate('window.LiveViewerDebug.snapshot().semanticCounts') == counts
            assert page.locator('#stampText').inner_text() == str(payload['stamp_ns']) + ' ns'
            report['checks'].append({'name': 'semantic_legend_ids_counts_palette_and_ns_timestamp_match', 'counts': counts})
            # Category hiding changes GPU visibility while leaving all uploaded
            # geometry and the independently saved traversability unchanged.
            classes = payload['semantic_registry']['classes']
            ceiling = next((item['id'] for item in classes if item['name'] == 'ceiling'), None)
            before_map = page.locator('#gridStats').inner_text()
            page.locator('#cutCeilingBtn').click()
            expected_visible = len(expected_labels) - int(np.count_nonzero(expected_labels == ceiling)) if ceiling is not None else len(expected_labels)
            assert page.evaluate('window.LiveViewerDebug.snapshot().visibleCount') == expected_visible
            assert page.locator('#gridStats').inner_text() == before_map
            report['checks'].append({'name': 'ceiling_cut_preserves_full_geometry_and_traversability', 'visible_voxels': expected_visible, 'stored_voxels': len(expected_labels)})
            preset_artifacts = page.evaluate('window.LiveViewerDebug.artifacts()')
            preset_instances = page.evaluate('window.LiveViewerDebug.instanceData()')
            for preset in ('official', 'oblique', 'top', 'low', 'side'):
                page.locator('#posePreset').select_option(preset)
                page.wait_for_timeout(100)
                state = page.evaluate('window.LiveViewerDebug.snapshot()')
                if preset == 'official':
                    assert state['projection'] == 'orthographic' and state['lighting'] == 'flat'
                if preset == 'oblique':
                    assert state['projection'] == 'perspective' and state['lighting'] == 'shaded'
                assert page.evaluate('window.LiveViewerDebug.artifacts()') == preset_artifacts
                assert page.evaluate('window.LiveViewerDebug.instanceData()') == preset_instances
                path = args.output / f'{args.run.name}_{args.frame:06d}_{preset}.png'
                page.locator('#scene').screenshot(path=str(path))
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert('RGB'))
                    assert int(np.count_nonzero(np.all(rgb == 255, axis=-1))) > 1000, 'White render background missing'
                    assert int(np.count_nonzero(np.any(rgb != 255, axis=-1))) > 1000, 'Actual 3D screenshot lacks a semantic voxel render'
                report['checks'].append({'name': 'real_webgl_semantic_white_screenshot', 'preset': preset,
                                         'projection': state['projection'], 'lighting': state['lighting'],
                                         'saved_geometry_labels_traversability_base_rgb_unchanged': True,
                                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'file': str(path)})
            view = page.evaluate('window.LiveViewerDebug.view()')
            assert view['geometry_modified'] is False and view['color_mode'] == 'semantic'
            page.evaluate('data=>window.LiveViewerDebug.applyView(data)', view)
            assert page.evaluate('window.LiveViewerDebug.view()') == view
            report['checks'].append({'name': 'view_json_round_trip', 'view': view})
            page.locator('#mode').select_option('current')
            current = api(url, args.frame, 'current')
            page.wait_for_function('n=>window.LiveViewerDebug.snapshot().mode === "current" && window.LiveViewerDebug.snapshot().cubeCount===n', arg=current['current_count'])
            assert page.locator('#gridStats').inner_text() == before_map
            for coloring in ('height', 'confidence', 'semantic'):
                page.locator('#colorMode').select_option(coloring)
                assert page.evaluate('window.LiveViewerDebug.snapshot().colorMode') == coloring
            report['checks'].append({'name': 'current_fused_and_all_color_controls_operate'})
            page.locator('#sourceToggle').click()
            page.wait_for_function('document.querySelectorAll("#sourceRows img").length>=4')
            page.wait_for_function('Array.from(document.querySelectorAll("#sourceRows img")).every(img=>img.complete && img.naturalWidth>0)', timeout=30000)
            report['checks'].append({'name': 'actual_synchronized_rgb_and_semantic_source_images_loaded',
                                     'images': page.locator('#sourceRows img').count()})
            with urlopen(f'{url}/api/sources?index={args.frame}', timeout=15) as response:
                source_payload = json.load(response)
            if source_payload['sources'] and all('expert' in item.get('semantic_variants', []) for item in source_payload['sources']):
                assert page.locator('#sourceSemanticControl').is_visible()
                before = page.evaluate('window.LiveViewerDebug.snapshot()')
                expert_colors = {0: [54, 94, 210], 1: [245, 155, 50], 2: [20, 166, 132], 255: [145, 145, 145]}
                expected_expert_totals = {str(key): 0 for key in expert_colors}
                for variant, field in [('official', 'official_labels'), ('indoor', 'b5_labels'), ('expert', 'expert_id'), ('final', 'labels')]:
                    page.locator('#sourceSemantic').select_option(variant)
                    kind = 'semantic' if variant == 'final' else variant
                    page.wait_for_function('''kind => {
                      const images=Array.from(document.querySelectorAll('#sourceRows img'));
                      return images.length===8 && images.every((img,i)=>img.complete && img.naturalWidth>0 &&
                        (i%2===0 || img.src.includes('kind='+kind)));
                    }''', arg=kind, timeout=30000)
                    assert page.evaluate('window.LiveViewerDebug.snapshot().sourceVariant') == variant
                    after = page.evaluate('window.LiveViewerDebug.snapshot()')
                    assert (after['cubeCount'], after['visibleCount'], after['mode'], after['colorMode']) == (before['cubeCount'], before['visibleCount'], before['mode'], before['colorMode'])
                    assert page.locator('#gridStats').inner_text() == before_map
                    image_checks = []
                    for item in source_payload['sources']:
                        data = load_semantic_image_arrays(args.run, args.frame, item)
                        labels = data[field]
                        source_colors = expert_colors if variant == 'expert' else palette
                        lookup = np.zeros((256, 3), np.uint8)
                        for cid, color in source_colors.items():
                            lookup[cid] = color
                        with urlopen(url + item['semantic_urls'][variant], timeout=15) as response:
                            encoded = response.read()
                        decoded = np.asarray(Image.open(io.BytesIO(encoded)).convert('RGB'))
                        np.testing.assert_array_equal(decoded, lookup[labels])
                        if variant == 'expert':
                            counts = {str(int(a)): int(b) for a, b in zip(*np.unique(labels, return_counts=True))}
                            assert item['expert_counts'] == counts
                            for key, count in counts.items():
                                expected_expert_totals[key] += count
                        image_checks.append({'view': item.get('view_id', item['camera']), 'pixels': int(labels.size), 'png_sha256': hashlib.sha256(encoded).hexdigest()})
                    if variant == 'expert':
                        assert page.locator('#sourceExpertLegend').is_visible()
                        legend = page.locator('#sourceExpertLegend').inner_text()
                        for key, count in expected_expert_totals.items():
                            assert f'{count:,}' in legend, (key, count, legend)
                    else:
                        assert not page.locator('#sourceExpertLegend').is_visible()
                    report['checks'].append({'name': 'actual_expert_source_pngs_and_controls_match_saved_arrays', 'variant': variant,
                                             'unchanged_3d_and_traversability': True, 'images': image_checks})
            else:
                assert not page.locator('#sourceSemanticControl').is_visible()
            page.screenshot(path=str(args.output / f'{args.run.name}_{args.frame:06d}_full_page.png'), full_page=True)
            browser.close()
    report['accepted'] = not report['javascript_errors'] and not report['console_errors']
    output = args.output / f'{args.run.name}_{args.frame:06d}_browser.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    assert report['accepted'], report
    print(json.dumps({'accepted': True, 'checks': len(report['checks']), 'report': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
