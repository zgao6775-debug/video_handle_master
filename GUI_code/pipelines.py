import os
import subprocess
import sys
import threading
import time


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def timestamp_id():
    return time.strftime("%Y%m%d_%H%M%S")


def run_subprocess(cmd, cwd=None, log=None):
    if log:
        log(" ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"未找到可执行程序：{cmd[0]}（请确认已安装并加入环境变量）") from e

    def reader():
        for line in proc.stdout:
            if log:
                log(line.rstrip("\n"))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    code = proc.wait()
    t.join(timeout=0.5)
    if code != 0:
        raise RuntimeError(f"命令执行失败，退出码 {code}")


def burn_subtitles_ffmpeg(video_path, srt_path, output_path, log=None):
    if not os.path.exists(srt_path) or os.path.getsize(srt_path) == 0:
        raise RuntimeError("字幕文件为空，无法压入（可能视频太短或未识别到语音）")
    srt_abs = os.path.abspath(srt_path).replace("\\", "/").replace(":", "\\:")
    vf_param = f"subtitles='{srt_abs}'"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        vf_param,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-c:a",
        "copy",
        output_path,
    ]
    run_subprocess(cmd, cwd=None, log=log)


def combine_video_audio_ffmpeg(video_path, audio_path, output_path, log=None):
    if not os.path.exists(audio_path):
        raise RuntimeError("未找到音频文件")
    ensure_dir(os.path.dirname(output_path))
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    run_subprocess(cmd, cwd=None, log=log)
    return output_path


def task_smart_bgm(video_path, out_dir, library_dir, volume=0.7, log=None, unified_dir=None, report_path=None, text_context=None):
    ensure_dir(out_dir)
    try:
        from Video_BGM.smart_bgm import smart_bgm
    except Exception as e:
        raise RuntimeError(f"智能配乐模块不可用: {e}") from e
    return smart_bgm(
        video_path,
        out_dir,
        library_dir=library_dir,
        volume=volume,
        log=log,
        unified_dir=unified_dir,
        report_path=report_path,
        text_context=text_context,
    )


def task_unified_report(video_path, out_dir, targets="en", whisper_model="small", with_srt=True, use_dashscope=False, log=None):
    root = repo_root()
    script = os.path.join(root, "video-unified-output", "unified_report.py")
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        script,
        "--input",
        video_path,
        "--outdir",
        out_dir,
        "--targets",
        targets,
        "--whisper-model",
        whisper_model,
    ]
    if with_srt:
        cmd.append("--with-srt")
    if use_dashscope:
        cmd.append("--use-dashscope")
    run_subprocess(cmd, cwd=root, log=log)
    return os.path.join(out_dir, "report.json")


def task_auto_effects(video_path, out_video, out_report, transition_fade=0.4, log=None):
    root = repo_root()
    script = os.path.join(root, "video-effects-add", "auto_effects.py")
    ensure_dir(os.path.dirname(out_video))
    cmd = [
        sys.executable,
        script,
        "--input",
        video_path,
        "--output",
        out_video,
        "--transition-fade",
        str(transition_fade),
        "--report",
        out_report,
    ]
    run_subprocess(cmd, cwd=os.path.join(root, "video-effects-add"), log=log)
    return out_video, out_report


def task_motion_analysis(video_path, out_report, max_frames=None, skip_model=False, no_display=True, log=None):
    root = repo_root()
    script = os.path.join(root, "video-Motion-Analysis", "main.py")
    ensure_dir(os.path.dirname(out_report))
    cmd = [
        sys.executable,
        script,
        "--source",
        video_path,
        "--report",
        out_report,
    ]
    if no_display:
        cmd.append("--no-display")
    if max_frames is not None:
        cmd.extend(["--max-frames", str(int(max_frames))])
    if skip_model:
        cmd.append("--skip-model")
    run_subprocess(cmd, cwd=os.path.join(root, "video-Motion-Analysis"), log=log)
    return out_report


def task_object_detection_yolo(video_path, out_dir, log=None):
    root = repo_root()
    cwd = os.path.join(root, "video-objects-detect")
    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, "yolo_out.avi")
    cmd = [sys.executable, "object_detection_yolo.py", "--video", video_path, "--output", out_file]
    run_subprocess(cmd, cwd=cwd, log=log)
    if os.path.exists(out_file):
        return out_file
    raise RuntimeError("未找到目标检测输出文件")


def task_burn_subtitles_from_unified(unified_out_dir, video_path, lang, out_video, log=None):
    report_path = os.path.join(unified_out_dir, "report.json")
    if not os.path.exists(report_path):
        raise RuntimeError("未找到 unified report.json，请先运行统一输出")
    import json

    report = json.load(open(report_path, "r", encoding="utf-8"))
    if lang == "auto":
        lang = ((report.get("transcript") or {}).get("language") or "").strip() or "zh"
    srt_map = (report.get("artifacts", {}).get("srt") or {})
    srt_path = srt_map.get(lang)
    if not srt_path or not os.path.exists(srt_path):
        available = [k for k, v in srt_map.items() if v and os.path.exists(v)]
        if available:
            raise RuntimeError(f"未找到 {lang} 字幕文件。当前可用字幕：{', '.join(available)}")
        raise RuntimeError("未找到字幕文件（SRT）。请先在“关键帧+摘要”里勾选“生成字幕文件（SRT）”。")
    ensure_dir(os.path.dirname(out_video))
    burn_subtitles_ffmpeg(video_path, srt_path, out_video, log=log)
    return out_video

