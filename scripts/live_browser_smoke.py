#!/usr/bin/env python3
"""Optional real-browser viewer acceptance, with no model inference.

Run with the isolated viewer test environment. This starts temporary read-only
localhost servers and uses real NPZ files from the completed recording. The
waiting-to-live fixture copies those files; it does not generate sensor data.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from urllib.request import urlopen

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parents[1]
PALETTE = [[83, 96, 118], [239, 139, 69], [234, 192, 84], [80, 189, 150]]


def read_api(url, index="latest", mode="fused"):
    with urlopen(f"{url}/api/frame?index={index}&mode={mode}", timeout=10) as response:
        return json.load(response)


def source_artifact(run, index, mode="fused"):
    with np.load(run / "occupancy" / f"{index:06d}.npz", allow_pickle=False) as artifact:
        size = float(artifact["voxel_size"])
        centers = artifact["bounds"][0] + (artifact[f"{mode}_indices"].astype(float) + 0.5) * size
        current_count = len(artifact["current_indices"])
        fused_count = len(artifact["fused_indices"])
    with np.load(run / "traversability" / f"{index:06d}.npz", allow_pickle=False) as artifact:
        states = artifact["traversability"].copy()
    return centers, states, current_count, fused_count


def check_payload(run, payload, index, mode):
    centers, states, current_count, fused_count = source_artifact(run, index, mode)
    assert payload["status"] == "ready", payload
    assert payload["index"] == index
    assert payload["mode"] == mode
    assert payload["current_count"] == current_count
    assert payload["fused_count"] == fused_count
    np.testing.assert_allclose(np.asarray(payload["centers"]), centers, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(np.asarray(payload["grid"]["traversability"]), states.ravel())
    return {"frame": index, "mode": mode, "cubes": len(centers),
            "map_class_counts": {str(code): int(np.count_nonzero(states == code)) for code in range(4)},
            "source_npz_sha256": hashlib.sha256((run / "occupancy" / f"{index:06d}.npz").read_bytes()).hexdigest()}


def load_backend():
    spec = importlib.util.spec_from_file_location("viewer_browser_backend", PROJECT / "scripts" / "live_viewer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_server(module, run, stack):
    server = module.make_server(run, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stack.callback(server.server_close)
    stack.callback(server.shutdown)
    return f"http://127.0.0.1:{server.server_port}"


def copy_frame(source, destination, index):
    for kind in ("occupancy", "traversability"):
        (destination / kind).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / kind / f"{index:06d}.npz", destination / kind / f"{index:06d}.npz")
    rows = (source / "frames.jsonl").read_text().splitlines()
    return rows[index]


def screenshot_check(page, output):
    page.screenshot(path=str(output), full_page=True)
    with Image.open(output) as image:
        assert image.width == 1400
        assert image.height >= 900
        assert len(image.getcolors(maxcolors=1_000_000)) > 100, "Screenshot is unexpectedly flat"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-run", type=Path, default=PROJECT / "runs" / "lidar_full")
    parser.add_argument("--camera-run", type=Path, default=PROJECT / "runs" / "camera_full")
    parser.add_argument("--output", type=Path, default=PROJECT / "diagnostics" / "live_viewer")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(PROJECT / ".viewer-test-browsers"))
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Real browser rendering and live-reader controls using actual completed NPZ files. No inference or occupancy accuracy assessment.",
              "checks": [], "javascript_errors": [], "console_errors": []}
    module = load_backend()
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix="ttocc-live-fixture-") as temp:
        lidar_url = start_server(module, args.lidar_run, stack)
        camera_url = start_server(module, args.camera_run, stack)
        fixture = Path(temp) / "not-created-yet"
        fixture_url = start_server(module, fixture, stack)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--enable-webgl", "--enable-unsafe-swiftshader",
                                               "--use-gl=angle", "--use-angle=swiftshader"])
            page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
            page.add_init_script("""(() => {
                window.__viewerNativeDraws = [];
                const native = WebGL2RenderingContext.prototype.drawArraysInstanced;
                WebGL2RenderingContext.prototype.drawArraysInstanced = function(mode, first, count, instances) {
                    native.call(this, mode, first, count, instances);
                    window.__viewerNativeDraws.push({mode, first, count, instances,
                        index: document.getElementById('frameNumber').value,
                        viewMode: document.getElementById('mode').value, glError: this.getError()});
                };
            })()""")
            page.on("pageerror", lambda error: report["javascript_errors"].append(str(error)))
            page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
            # DOM checks are intentionally implemented below against the viewer's
            # stable diagnostics, and require a real WebGL2 context.
            try:
                run_browser_checks(page, args, report, lidar_url, camera_url, fixture_url, fixture)
            except Exception as error:
                report["accepted"] = False
                report["failure"] = f"{type(error).__name__}: {error}"
                (args.output / "browser_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                raise
            finally:
                browser.close()
    report["accepted"] = not report["javascript_errors"] and not report["console_errors"]
    assert report["accepted"], report
    (args.output / "browser_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"accepted": True, "checks": len(report["checks"]), "report": str(args.output / "browser_report.json")}, ensure_ascii=False))


def run_browser_checks(page, args, report, lidar_url, camera_url, fixture_url, fixture):
    def wait_frame(index, cubes, mode="fused"):
        page.wait_for_function("""expect => {
            const state = window.LiveViewerDebug?.snapshot();
            const draw = window.__viewerNativeDraws?.at(-1);
            return state && state.index === expect.index && state.mode === expect.mode &&
                state.cubeCount === expect.cubes && state.webgl2 && draw &&
                draw.index === String(expect.index) && draw.viewMode === expect.mode &&
                draw.instances === expect.cubes && draw.count === 36 && draw.mode === 4 && draw.glError === 0;
        }""", arg={"index": index, "cubes": cubes, "mode": mode}, timeout=20000)
        return page.evaluate("window.LiveViewerDebug.snapshot()")

    def check_map(run, index):
        _, states, _, _ = source_artifact(run, index)
        samples = []
        for code in range(4):
            coords = np.argwhere(states == code)
            # Skip border strokes, axis labels, and the sensor-origin cross.
            coords = [tuple(map(int, point)) for point in coords
                      if np.all(point > 2) and np.all(point < np.array(states.shape) - 3)
                      and np.any(np.abs(point - np.array(states.shape) / 2) > 4)]
            for point in coords[::max(1, len(coords) // 3)][:3]:
                samples.append({"x": point[0], "y": point[1], "code": code})
        pixels = page.evaluate("""({samples, shape}) => {
            const canvas = document.getElementById('map'), context = canvas.getContext('2d');
            const dpr = Math.min(window.devicePixelRatio || 1, 2), [nx, ny] = shape;
            const scale = Math.max(.01, Math.min((canvas.width - 54*dpr)/ny, (canvas.height - 54*dpr)/nx));
            const ox = (canvas.width-ny*scale)/2, oy = (canvas.height-nx*scale)/2;
            return samples.map(p => ({...p, rgba:Array.from(context.getImageData(
                Math.floor(ox+(ny-1-p.y+.5)*scale), Math.floor(oy+(nx-1-p.x+.5)*scale), 1, 1).data)}));
        }""", {"samples": samples, "shape": list(states.shape)})
        for pixel in pixels:
            assert pixel["rgba"] == PALETTE[pixel["code"]] + [255], pixel
        assert len(pixels) >= 3
        return {"actual_canvas_pixels_checked": pixels, "classes_absent_from_source":
                [code for code in range(4) if not np.any(states == code)]}

    def capture_scene(path):
        page.locator("#scene").screenshot(path=str(path))
        with Image.open(path) as image:
            colors = image.convert("RGB").getcolors(maxcolors=1_000_000)
            assert colors is not None and len(colors) > 50
            colored_pixels = sum(n for n, color in colors if max(color) - min(color) > 35 and max(color) > 75)
            assert colored_pixels > 1000, "No substantial colored voxel raster in real WebGL screenshot"
            return {"scene_unique_colors": len(colors), "scene_colored_pixels": colored_pixels,
                    "scene_png_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    for route, run, url in (("lidar", args.lidar_run, lidar_url), ("camera", args.camera_run, camera_url)):
        latest = read_api(url)
        assert latest["processed_frames"] == 2348 and latest["complete"] is True
        record = check_payload(run, latest, 2347, "fused")
        page.goto(url, wait_until="domcontentloaded")
        state = wait_frame(2347, len(latest["centers"]))
        assert state["follow"] is True and state["glError"] == 0
        assert page.locator("#stampText").inner_text() == str(latest["stamp_ns"]) + " ns", "Nanosecond timestamp lost integer precision"
        assert "全包处理完成" in page.locator("#statusText").inner_text()
        record.update({"check": f"{route}_latest_actual_npz_real_webgl2", "browser_state": state})
        record["native_gpu_context"] = page.evaluate("""() => {
            const gl = document.getElementById('scene').getContext('webgl2');
            const extension = gl.getExtension('WEBGL_debug_renderer_info');
            return {version:gl.getParameter(gl.VERSION), vendor:extension?gl.getParameter(extension.UNMASKED_VENDOR_WEBGL):null,
                renderer:extension?gl.getParameter(extension.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER)};
        }""")
        record.update(check_map(run, 2347))
        record.update(capture_scene(args.output / f"{route}_latest_scene.png"))
        screenshot_check(page, args.output / f"{route}_latest.png")
        report["checks"].append(record)
        page.locator("#followBtn").click()
        assert page.locator("#followBtn").get_attribute("aria-pressed") == "false"
        current = read_api(url, mode="current")
        record = check_payload(run, current, 2347, "current")
        page.locator("#mode").select_option("current")
        wait_frame(2347, len(current["centers"]), "current")
        assert current["current_count"] < latest["fused_count"], "Selected real frame should show temporal fusion contribution"
        record.update({"check": f"{route}_paused_current_mode_actual_voxel_count", "map": check_map(run, 2347)})
        report["checks"].append(record)
        page.locator("#mode").select_option("fused")
        wait_frame(2347, len(latest["centers"]))
        zero = read_api(url, index=0)
        record = check_payload(run, zero, 0, "fused")
        page.locator("#frameSlider").evaluate("element => {element.value='0';element.dispatchEvent(new Event('input',{bubbles:true}));}")
        wait_frame(0, len(zero["centers"]))
        assert page.locator("#followBtn").get_attribute("aria-pressed") == "false"
        record.update({"check": f"{route}_historical_slider_zero", "map": check_map(run, 0)})
        screenshot_check(page, args.output / f"{route}_frame0.png")
        report["checks"].append(record)
        centers, _, _, _ = source_artifact(run, 0)
        selected_count = int(np.count_nonzero((centers[:, 2] >= 0) & (centers[:, 2] <= 2)))
        full_scene = page.locator("#scene").screenshot()
        page.locator("#clipEnabled").check()
        page.locator("#zMin").fill("0")
        page.locator("#zMax").fill("2")
        page.wait_for_function("expected => Number(document.getElementById('visibleValue').textContent.split('/')[0].replaceAll(',','').trim()) === expected", arg=selected_count)
        page.wait_for_timeout(100)
        assert page.locator("#scene").screenshot() != full_scene
        assert page.evaluate("window.LiveViewerDebug.snapshot().cubeCount") == len(centers)
        page.locator("#clipEnabled").uncheck()
        report["checks"].append({"check": f"{route}_height_clip_actual_centers", "visible_centers": selected_count,
                                 "source_centers": len(centers), "source_npz_not_modified": True})
        # Native pointer controls must change real raster content, then reset.
        before = page.locator("#scene").screenshot()
        scene = page.locator("#scene").bounding_box()
        page.mouse.move(scene["x"] + scene["width"] * .5, scene["y"] + scene["height"] * .5)
        page.mouse.down()
        page.mouse.move(scene["x"] + scene["width"] * .62, scene["y"] + scene["height"] * .52, steps=10)
        page.mouse.up()
        page.wait_for_timeout(100)
        after = page.locator("#scene").screenshot()
        assert before != after
        page.locator("#resetBtn").click()
        page.wait_for_timeout(100)
        report["checks"].append({"check": f"{route}_drag_changes_real_scene_raster", "different_png_bytes": True})
        page.locator("#followBtn").click()
        wait_frame(2347, len(latest["centers"]))
        report["checks"].append({"check": f"{route}_resume_follows_latest", "index": 2347})

    assert read_api(fixture_url)["status"] == "waiting"
    page.goto(fixture_url, wait_until="domcontentloaded")
    page.wait_for_function("document.getElementById('statusText').textContent.includes('等待')")
    screenshot_check(page, args.output / "waiting.png")
    assert page.evaluate("window.LiveViewerDebug.snapshot().index") is None
    fixture.mkdir()
    shutil.copy2(args.lidar_run / "run.json", fixture / "run.json")
    row0 = copy_frame(args.lidar_run, fixture, 0)
    (fixture / "frames.jsonl").write_text(row0)
    assert read_api(fixture_url)["status"] == "waiting", "Unterminated journal must not display copied artifacts"
    page.wait_for_timeout(600)
    assert page.evaluate("window.LiveViewerDebug.snapshot().index") is None
    with (fixture / "frames.jsonl").open("a") as journal:
        journal.write("\n")
    live0 = read_api(fixture_url)
    record = check_payload(fixture, live0, 0, "fused")
    wait_frame(0, len(live0["centers"]))
    assert live0["complete"] is False
    record.update({"check": "missing_run_waits_then_follows_real_committed_frame", "unterminated_commit_ignored": True})
    screenshot_check(page, args.output / "live_fixture_frame0.png")
    report["checks"].append(record)
    page.locator("#followBtn").click()
    row1 = copy_frame(args.lidar_run, fixture, 1)
    with (fixture / "frames.jsonl").open("a") as journal:
        journal.write(row1 + "\n")
    live1 = read_api(fixture_url)
    record = check_payload(fixture, live1, 1, "fused")
    page.wait_for_timeout(1100)
    assert page.evaluate("window.LiveViewerDebug.snapshot().index") == 0
    page.locator("#followBtn").click()
    wait_frame(1, len(live1["centers"]))
    assert page.locator("#followBtn").get_attribute("aria-pressed") == "true"
    record.update({"check": "pause_retains_previous_live_frame_resume_reads_new_commit", "paused_index": 0, "resumed_index": 1})
    report["checks"].append(record)
    draws = page.evaluate("window.__viewerNativeDraws")
    assert draws and all(draw["count"] == 36 and draw["mode"] == 4 and draw["glError"] == 0 for draw in draws)
    report["checks"].append({"check": "native_webgl_instanced_cube_draws", "actual_calls": draws})


if __name__ == "__main__":
    main()
