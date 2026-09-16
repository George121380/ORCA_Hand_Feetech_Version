"""Browser-guided Mac camera capture, followed by the real ORCA RGB pipeline.

Run: .venv/bin/python scripts/webcam_validation.py
Open the printed localhost URL and follow the 40-second capture guide.
Camera recordings and reports stay under the ignored runs/ directory.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASES = [
    {"name": "张开右手", "hint": "手掌朝镜头，五指自然展开，保持稳定完成尺度标定。", "seconds": 8},
    {"name": "缓慢握拳", "hint": "缓慢合拢手指，然后保持握拳。", "seconds": 6},
    {"name": "只伸食指", "hint": "食指伸直，其余手指弯曲，保持不动。", "seconds": 6},
    {"name": "竖起拇指", "hint": "做一个点赞手势，拇指伸直，其余手指弯曲。", "seconds": 6},
    {"name": "拇指与食指捏合", "hint": "拇指和食指指尖轻触，其他手指保持自然。", "seconds": 6},
    {"name": "把手移出画面", "hint": "让摄像头看不到手，用来验证丢手后的保持行为。", "seconds": 3},
    {"name": "重新张开右手", "hint": "把手移回画面，五指展开，观察是否恢复跟踪。", "seconds": 5},
]


def analyze(folder: Path, state: dict):
    """Separate process for ML/rendering; request threads only handle capture UI."""
    try:
        import imageio_ffmpeg
        import numpy as np
        from orca_feetech.model import JOINT_NAMES
        with (folder / "analysis.log").open("w") as log:
            recording = folder / ("capture.mp4" if (folder / "capture.mp4").exists() else "capture.webm")
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(recording),
                            "-vf", "fps=20,scale=640:-2", "-an", "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(folder / "input.mp4")],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
            state["message"] = "正在检测人手并驱动仿真，处理时间通常接近视频时长。"
            subprocess.run([sys.executable, "-m", "orca_feetech.cli", "teleop", "--source",
                            str(folder / "input.mp4"), "--headless", "--video", "--output", str(folder / "analysis")],
                           check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(folder / "analysis/rgb_overlay.mp4"),
                            "-i", str(folder / "analysis/simulation.mp4"), "-filter_complex",
                            "[0:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2[a];[a][1:v]hstack=inputs=2[v]",
                            "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(folder / "comparison.mp4")],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        rows = [json.loads(x) for x in (folder / "analysis/trajectory.jsonl").read_text().splitlines()]
        metrics = json.loads((folder / "analysis/vision_metrics.json").read_text())
        capture = json.loads((folder / "capture.json").read_text())
        landmarks = [json.loads(x) for x in (folder / "analysis/landmarks.jsonl").read_text().splitlines()]
        source_duration = rows[-1]["time_s"] - rows[0]["time_s"] if len(rows) > 1 else 0
        phases, start = [], 0
        for phase in PHASES:
            end = start + phase["seconds"]
            segment = [r for r in rows if start <= r["time_s"] < end]
            detected = sum(start <= r["time_s"] < end for r in landmarks)
            if segment:
                tail = [r for r in segment if r["time_s"] >= end - 2 and r["input_valid"]]
                summary = {**phase, "start_s": start, "end_s": end, "frames": len(segment),
                           "detected_frames": detected, "commanded_frames": sum(r["input_valid"] for r in segment)}
                if tail:
                    q = np.rad2deg(np.median([r["target"] for r in tail], axis=0))
                    summary["last_two_seconds_target_deg"] = dict(zip(JOINT_NAMES, q.round(2).tolist()))
                phases.append(summary)
            start = end
        gaps = [(a, b) for a, b in zip(rows, rows[1:]) if not b["input_valid"]]
        metrics.update({"source_kind": capture.get("source", "unknown_video"), "capture": capture,
                        "video_duration_s": source_duration, "phases": phases,
                        "invalid_transitions": len(gaps),
                        "invalid_commands_held": all(a["command"] == b["command"] for a, b in gaps),
                        "scope": "Recorded video processed offline; not live inference latency or hand-pose ground truth"})
        from review_webcam import review
        metrics["review"] = review(folder, metrics)
        (folder / "validation.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
        report(folder, metrics)
        state.update(status="complete", message="处理完成。现在可以查看真人与仿真对照。", report="/report.html")
        print(json.dumps({"status": "complete", "output": str(folder), "frames": len(rows),
                          "detected_frames": metrics["detected_frames"], "commanded_frames": metrics["commanded_frames"]}), flush=True)
    except Exception as exc:
        state.update(status="failed", message=f"分析失败：{exc}。详情见 analysis.log。")
        print(state["message"], flush=True)


def report(folder, metrics):
    real_camera = metrics["source_kind"] in {"mac_camera_via_browser", "mac_camera_native"}
    source_label = "真实摄像头录像" if real_camera else "软件测试素材"
    rows = "".join(f'<tr><td>{html.escape(p["name"])}</td><td>{p["start_s"]}–{p["end_s"]} s</td>'
                   f'<td>{p["detected_frames"]} / {p["frames"]}</td><td>{p["commanded_frames"]}</td></tr>'
                   for p in metrics["phases"])
    findings = []
    review = metrics.get("review", {})
    interval = review.get("main_guided_interval", {})
    if interval.get("frames"):
        findings.append(f'8–32 秒主要手势阶段：{interval["frames"]} 帧中检测到右手 '
                        f'{interval["detected_frames"]} 帧，生成有效目标 {interval["commanded_frames"]} 帧。')
    if review.get("first_detected_s") is not None:
        first_command = review.get("first_commanded_s")
        findings.append(f'首次检测到右手：{review["first_detected_s"]:.2f} 秒；'
                        + (f'完成尺度标定并开始输出目标：{first_command:.2f} 秒。'
                           if first_command is not None else '尚未完成尺度标定。'))
    for gap in review.get("post_calibration_dropouts", []):
        resume = f'{gap["resumed_s"]:.2f} 秒恢复' if gap["resumed_s"] is not None else '至录像结束未恢复'
        findings.append(f'丢手：{gap["start_s"]:.2f}–{gap["last_invalid_s"]:.2f} 秒，'
                        f'{gap["frames"]} 帧命令保持{ "通过" if gap["commands_held"] else "未通过"}；{resume}。')
    findings.extend(metrics.get("visual_review", []))
    review_html = '<h2>检查结论</h2><ul class="findings">' + ''.join(
        f'<li>{html.escape(finding)}</li>' for finding in findings) + '</ul>' if findings else ''
    kinematics = review.get("phase_kinematics", [])
    pinch = next((p for p in kinematics if p["name"] == "拇指与食指捏合"), None)
    if pinch:
        review_html += (
            '<div class="note warning"><strong>捏合：指尖距离需要继续检查</strong><br>'
            f'{pinch["tail_start_s"]}–{pinch["tail_end_s"]} 秒，拇指与食指参考点间距中位数：'
            f'重定向目标 <strong>{pinch["target_thumb_index_tip_gap_median_mm"]:.1f} mm</strong>，'
            f'仿真反馈 <strong>{pinch["sim_thumb_index_tip_gap_median_mm"]:.1f} mm</strong>。'
            f'该段 16 个手指关节的目标与仿真反馈平均绝对误差为 '
            f'{pinch["sim_target_to_feedback_mae_deg"]:.2f}°。'
            '<br><small>间距由官方 v1 URDF 指尖坐标原点计算，不能当作碰撞表面的接触间隙。</small></div>'
        )
    if kinematics:
        kine_rows = ''.join(
            f'<tr><td>{html.escape(p["name"])}</td><td>{p["tail_start_s"]}–{p["tail_end_s"]} s</td>'
            f'<td>{p["valid_samples"]}</td><td>{p["sim_target_to_feedback_mae_deg"]:.2f}°</td></tr>'
            for p in kinematics)
        review_html += ('<details><summary>查看各动作末段的仿真跟踪误差</summary>'
                        '<p class="muted">每个提示阶段最后 2 秒的有效帧，统计 16 个手指关节的目标与仿真反馈平均绝对误差。这不是人手姿态误差。</p>'
                        '<div class="table-wrap"><table><thead><tr><th>提示动作</th><th>采样时间</th><th>有效帧</th><th>平均绝对误差</th></tr></thead>'
                        f'<tbody>{kine_rows}</tbody></table></div></details>')
    duration = metrics.get("capture", {}).get("duration_s", metrics["video_duration_s"])
    stats = [(f'{duration:.1f} s', '录像时长'), (metrics['frames'], '处理帧'),
             (metrics['detected_frames'], '检测到右手'), (metrics['commanded_frames'], '有效控制目标')]
    seeks = ''.join(f'<button type="button" data-seek="{min(p["end_s"] - 1.5, duration - 0.05):.2f}">{html.escape(p["name"])}</button>'
                    for p in metrics['phases'] if p['start_s'] < duration)
    gaps = review.get("post_calibration_dropouts", [])
    seeks += ''.join(f'<button type="button" data-seek="{(p["start_s"] + p["last_invalid_s"]) / 2:.2f}">丢手保持</button>' for p in gaps)
    values = {
        "SOURCE": source_label, "PHASES": rows, "SEEKS": seeks, "REVIEW": review_html,
        "STATS": ''.join(f'<div><div class="stat-value">{value}</div>{label}</div>' for value, label in stats),
        "HOLD": f'全部无效输入采样的命令保持检查：{"通过" if metrics["invalid_commands_held"] and metrics["invalid_transitions"] else "未覆盖或未通过"}。包含开始时的等待与标定；实际丢手时段见上方结论。',
        "P50": f'{metrics["processing_p50_ms"]:.1f}', "P95": f'{metrics["processing_p95_ms"]:.1f}',
        "FOLDER": html.escape(str(folder)),
    }
    template = (ROOT / "scripts/webcam_report.template.html").read_text()
    for name, value in values.items():
        template = template.replace(f"__{name}__", value)
    (folder / "report.html").write_text(template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output", default=f"runs/webcam-{datetime.now():%Y%m%d-%H%M%S}")
    parser.add_argument("--serve-existing", action="store_true", help="Serve a completed report without opening the camera")
    args = parser.parse_args()
    folder = (ROOT / args.output).resolve()
    if args.serve_existing:
        if not (folder / "report.html").is_file():
            parser.error("--serve-existing requires an output folder containing report.html")
    else:
        folder.mkdir(parents=True, exist_ok=False)
    state = {"status": "ready", "message": "准备摄像头", "output": str(folder)}
    if args.serve_existing:
        state.update(status="complete", message="已载入完成的验证报告。", report="/report.html")
    camera = [None]
    upload_lock = threading.Lock()
    template = (ROOT / "scripts/webcam_validation.html").read_text().replace("__PHASES__", json.dumps(PHASES, ensure_ascii=False))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, payload, content_type="application/json", code=200, headers=None):
            if isinstance(payload, dict):
                payload = json.dumps(payload, ensure_ascii=False).encode()
            elif isinstance(payload, str):
                payload = payload.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def send_file(self, path, content_type):
            # Chrome needs byte ranges to seek in MP4 files before fully loading them.
            size = path.stat().st_size
            headers = {"Accept-Ranges": "bytes"}
            requested = self.headers.get("Range")
            if not requested:
                return self.send(path.read_bytes(), content_type, headers=headers)
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if match and any(match.groups()):
                left, right = match.groups()
                start = int(left) if left else max(0, size - int(right))
                end = min(int(right), size - 1) if left and right else size - 1
                if 0 <= start <= end < size:
                    with path.open("rb") as stream:
                        stream.seek(start)
                        data = stream.read(end - start + 1)
                    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
                    return self.send(data, content_type, code=206, headers=headers)
            headers["Content-Range"] = f"bytes */{size}"
            return self.send(b"", content_type, code=416, headers=headers)

        def do_GET(self):
            if self.path == "/":
                if args.serve_existing:
                    return self.send_file(folder / "report.html", "text/html; charset=utf-8")
                return self.send(template, "text/html; charset=utf-8")
            if self.path == "/api/state":
                snapshot = dict(state)
                trajectory = folder / "analysis/trajectory.jsonl"
                if state["status"] == "analyzing" and trajectory.exists():
                    snapshot["processed_frames"] = len(trajectory.read_text().splitlines())
                return self.send(snapshot)
            if self.path.startswith("/api/camera/frame"):
                if camera[0] is not None and camera[0].jpeg:
                    return self.send(camera[0].jpeg, "image/jpeg")
                return self.send({"error": "Camera is not open"}, code=404)
            allowed = {"/report.html": "text/html; charset=utf-8", "/comparison.mp4": "video/mp4",
                       "/input.mp4": "video/mp4", "/validation.json": "application/json",
                       "/analysis/tracking.png": "image/png"}
            if self.path in allowed and (folder / self.path[1:]).is_file():
                return self.send_file(folder / self.path[1:], allowed[self.path])
            self.send({"error": "Not found"}, code=404)

        def do_POST(self):
            if args.serve_existing:
                return self.send({"error": "This session serves a completed report"}, code=409)
            if self.headers.get("Origin") not in {f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"}:
                return self.send({"error": "Origin rejected"}, code=403)
            if self.path.startswith("/api/camera/"):
                try:
                    if self.path == "/api/camera/open":
                        from orca_feetech.camera_capture import LocalCamera
                        with upload_lock:
                            if camera[0] is None:
                                camera[0] = LocalCamera()
                        h, w = camera[0].frame.shape[:2]
                        return self.send({"ready": True, "width": w, "height": h})
                    if self.path == "/api/camera/start":
                        with upload_lock:
                            if camera[0] is None or state["status"] != "ready":
                                return self.send({"error": "Camera/session is not ready"}, code=409)
                            state.update(status="recording", message="正在录制真实摄像头画面。")
                            def finished(error):
                                if error:
                                    state.update(status="failed", message=error)
                                else:
                                    state.update(status="analyzing", message="录制已保存，正在分析。")
                                    analyze(folder, state)
                            camera[0].start(folder, sum(p["seconds"] for p in PHASES), PHASES, finished)
                        return self.send({"recording": True})
                    if self.path == "/api/camera/stop":
                        if camera[0] is not None:
                            camera[0].stop_recording.set()
                        return self.send({"stopping": True})
                    return self.send({"error": "Not found"}, code=404)
                except (RuntimeError, ValueError, OSError) as exc:
                    return self.send({"error": str(exc)}, code=400)
            if self.path != "/api/recording":
                return self.send({"error": "Not found"}, code=404)
            size = int(self.headers.get("Content-Length", 0))
            if not 1000 < size < 100_000_000:
                return self.send({"error": "Invalid recording size"}, code=400)
            with upload_lock:
                if state["status"] != "ready":
                    return self.send({"error": "This session already has a recording"}, code=409)
                try:
                    metadata = json.loads(self.headers.get("X-Capture-Metadata", "{}"))
                    (folder / "capture.webm").write_bytes(self.rfile.read(size))
                    metadata.update({"recorded_at": datetime.now().astimezone().isoformat(), "phases": PHASES})
                    (folder / "capture.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
                    state.update(status="analyzing", message="录制已保存，正在转换视频。")
                    threading.Thread(target=analyze, args=(folder, state), daemon=True).start()
                    self.send({"accepted": True})
                except (ValueError, OSError) as exc:
                    self.send({"error": str(exc)}, code=400)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Camera validation: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Recordings: {folder}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if camera[0] is not None:
            camera[0].closed.set()
            camera[0].stop_recording.set()
        server.server_close()


if __name__ == "__main__":
    main()
