import argparse
import json
import os
import re
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import cv2
import numpy as np


LANG_NORMALIZE = {
    "zh-cn": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "chinese": "zh",
    "jp": "ja",
}

LANG_DISPLAY_ZH = {
    "zh": "中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "fr": "法语",
    "es": "西班牙语",
    "de": "德语",
    "ru": "俄语",
}


def _norm_lang(lang):
    if not lang:
        return ""
    l = lang.strip().lower()
    l = LANG_NORMALIZE.get(l, l)
    if l.startswith("zh"):
        return "zh"
    return l


def _ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = (total / fps) if fps else 0.0
    return {"path": video_path, "fps": fps, "total_frames": total, "width": w, "height": h, "duration_sec": duration}


def _detect_scenes(video_path, step=12, hist_thresh=0.45, motion_thresh=12.0):
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    prev_hist = None
    prev_gray = None
    cuts = [0]
    motion_values = []
    brightness_values = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, None).flatten()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append((idx, float(np.mean(gray))))
            if prev_hist is not None:
                d = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                diff = cv2.absdiff(gray, prev_gray)
                md = float(np.mean(diff))
                motion_values.append((idx, md))
                if d > hist_thresh:
                    cuts.append(idx)
            prev_hist = hist
            prev_gray = gray
        idx += 1
    cap.release()
    cuts.append(total)

    segments = []
    for i in range(len(cuts) - 1):
        start = int(cuts[i])
        end = int(cuts[i + 1])
        seg_motion = [m for f, m in motion_values if f >= start and f < end]
        avg_motion = float(np.mean(seg_motion)) if seg_motion else 0.0
        seg_bright = [b for f, b in brightness_values if f >= start and f < end]
        avg_brightness = float(np.mean(seg_bright)) if seg_bright else 0.0
        level = "high" if avg_motion >= motion_thresh else "low"
        segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_time": (start / fps) if fps else 0.0,
                "end_time": (end / fps) if fps else 0.0,
                "avg_motion": avg_motion,
                "avg_brightness": avg_brightness,
                "level": level,
            }
        )
    return segments


def _extract_keyframes(video_path, out_dir, max_keyframes=12, min_gap_sec=1.0):
    info = _video_info(video_path)
    fps = info["fps"]
    if not fps:
        raise RuntimeError("无法读取FPS")

    segments = _detect_scenes(video_path)
    target_frames = []
    for seg in segments:
        target_frames.append(int(seg["start_frame"]))
        target_frames.append(int((seg["start_frame"] + seg["end_frame"]) / 2))

    target_frames = sorted(set([f for f in target_frames if f >= 0 and f < info["total_frames"]]))
    min_gap_frames = int(min_gap_sec * fps)
    filtered = []
    last = -10**9
    for f in target_frames:
        if f - last >= min_gap_frames:
            filtered.append(f)
            last = f

    if len(filtered) > max_keyframes:
        idxs = np.linspace(0, len(filtered) - 1, num=max_keyframes).round().astype(int).tolist()
        filtered = [filtered[i] for i in idxs]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    _ensure_dir(out_dir)
    keyframes = []
    for i, frame_idx in enumerate(filtered, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        t = frame_idx / fps
        filename = f"keyframe_{i:03d}_{t:0.2f}s.jpg"
        out_path = os.path.join(out_dir, filename)
        cv2.imwrite(out_path, frame)
        keyframes.append(
            {
                "index": i,
                "frame": int(frame_idx),
                "time": float(t),
                "path": out_path,
            }
        )
    cap.release()
    return keyframes, segments


def _load_whisper():
    try:
        import whisper

        return whisper
    except Exception:
        return None


def _find_dashscope_key():
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_KEY")
    if key:
        return key.strip()
    cfg = os.path.join(os.path.dirname(__file__), "..", "Video-auto-editing", "config.py")
    cfg = os.path.abspath(cfg)
    if os.path.exists(cfg):
        txt = open(cfg, "r", encoding="utf-8", errors="ignore").read()
        m = re.search(r"API_KEY\s*=\s*['\"]([^'\"]+)['\"]", txt)
        if m:
            k = m.group(1).strip()
            if k and "YOUR" not in k.upper():
                return k
    return None


def _load_dashscope():
    try:
        import dashscope

        return dashscope
    except Exception:
        return None


def _transcribe(video_path, model_name="small", language=None, task="transcribe", device=None):
    whisper = _load_whisper()
    if whisper is None:
        return None
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception:
        return None
    kwargs = {"verbose": False}
    if language and language != "auto":
        kwargs["language"] = language
    if task:
        kwargs["task"] = task
    try:
        result = model.transcribe(video_path, **kwargs)
    except Exception:
        return None
    segments = []
    for seg in result.get("segments", []):
        segments.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or "").strip(),
            }
        )
    detected_lang = _norm_lang(result.get("language"))
    full_text = (result.get("text") or "").strip()
    return {"language": detected_lang, "text": full_text, "segments": segments}


def _write_srt(segments, out_path):
    def fmt(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        ms = int((s - int(s)) * 1000)
        return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"

    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{fmt(seg['start'])} --> {fmt(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")


def _simple_summary(text, max_chars=900):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return ""
    return text[:max_chars]


def _dashscope_chat(api_key, system, user, model="qwen-max-latest"):
    dashscope = _load_dashscope()
    if dashscope is None:
        return None
    def _call():
        return dashscope.Generation.call(
            api_key=api_key,
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            result_format="message",
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call)
            resp = fut.result(timeout=15)
        if getattr(resp, "status_code", None) != 200:
            return None
        return resp.output.choices[0].message.content
    except (FuturesTimeoutError, Exception):
        return None


def _summarize(transcript_text, source_lang, api_key=None, use_dashscope=False):
    if api_key and use_dashscope:
        system = "你是一个视频内容摘要助手。输出简洁要点和一句话摘要。"
        src_name = LANG_DISPLAY_ZH.get(_norm_lang(source_lang), source_lang)
        user = f"请根据以下转写内容生成摘要（保留原语言）。\n\n语言: {src_name}\n\n内容:\n{transcript_text}"
        out = _dashscope_chat(api_key, system, user)
        if out:
            return out.strip()
    return _simple_summary(transcript_text)


def _translate(text, source_lang, target_lang, api_key=None, use_dashscope=False):
    text = (text or "").strip()
    if not text:
        return ""
    src = _norm_lang(source_lang)
    tgt = _norm_lang(target_lang)
    if src == tgt:
        return text
    if api_key and use_dashscope:
        system = "你是一个高质量翻译引擎，只输出翻译结果，不要解释。"
        src_name = LANG_DISPLAY_ZH.get(src, src or source_lang)
        tgt_name = LANG_DISPLAY_ZH.get(tgt, tgt or target_lang)
        user = f"把下面内容从{src_name}翻译成{tgt_name}：\n\n{text}"
        out = _dashscope_chat(api_key, system, user)
        if out:
            return out.strip()
    return ""


def build_report(video_path, out_dir, targets, whisper_model, max_keyframes, min_gap, with_srt, use_dashscope):
    info = _video_info(video_path)
    frames_dir = os.path.join(out_dir, "keyframes")
    keyframes, scenes = _extract_keyframes(video_path, frames_dir, max_keyframes=max_keyframes, min_gap_sec=min_gap)

    api_key = _find_dashscope_key()
    transcript_src = _transcribe(video_path, model_name=whisper_model, language="auto", task="transcribe")
    source_lang = _norm_lang((transcript_src or {}).get("language") or "unknown")

    transcript_en = None
    if "en" in targets:
        transcript_en = _transcribe(video_path, model_name=whisper_model, language=None, task="translate")

    summary_src = _summarize((transcript_src or {}).get("text", ""), source_lang, api_key=api_key, use_dashscope=use_dashscope)
    summaries = {source_lang: summary_src} if summary_src else {}

    translations = {}
    for lang in targets:
        lang = _norm_lang(lang)
        if not lang or lang == source_lang:
            continue
        if lang == "en" and transcript_en and transcript_en.get("text"):
            translations["en"] = {"transcript": transcript_en["text"]}
            if "en" not in summaries:
                summaries["en"] = _summarize(transcript_en["text"], "en", api_key=api_key, use_dashscope=use_dashscope)
            continue
        t = _translate(summary_src, source_lang, lang, api_key=api_key, use_dashscope=use_dashscope)
        if t:
            summaries[lang] = t

    artifacts = {}
    if with_srt and transcript_src is not None:
        srt_path = os.path.join(out_dir, "subtitles", f"subtitle_{source_lang}.srt")
        _write_srt((transcript_src.get("segments") or []), srt_path)
        artifacts["srt"] = {source_lang: srt_path}
        if transcript_en is not None:
            srt_en = os.path.join(out_dir, "subtitles", "subtitle_en.srt")
            _write_srt((transcript_en.get("segments") or []), srt_en)
            artifacts["srt"]["en"] = srt_en

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video": info,
        "keyframes": keyframes,
        "scenes": scenes,
        "transcript": {"language": source_lang, "text": (transcript_src or {}).get("text", ""), "segments": (transcript_src or {}).get("segments", [])},
        "summaries": summaries,
        "translations": translations,
        "artifacts": artifacts,
        "notes": {
            "translation_backend": ("dashscope" if (api_key and use_dashscope) else ("whisper_translate_en" if transcript_en else "none")),
            "dashscope_key_found": bool(api_key),
            "dashscope_enabled": bool(use_dashscope),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="关键帧与摘要统一输出（支持多语言/翻译）")
    parser.add_argument("--input", required=True, type=str, help="输入视频路径")
    parser.add_argument("--outdir", default=None, type=str, help="输出目录（默认与视频同级的 unified_output）")
    parser.add_argument("--targets", default="en", type=str, help="目标语言列表，逗号分隔，如 en,zh,ja")
    parser.add_argument("--whisper-model", default="small", type=str, help="Whisper模型: tiny/base/small/medium/large")
    parser.add_argument("--max-keyframes", default=12, type=int, help="最多输出关键帧数量")
    parser.add_argument("--min-gap", default=1.0, type=float, help="关键帧最小间隔(秒)")
    parser.add_argument("--with-srt", action="store_true", help="同时输出SRT字幕文件")
    parser.add_argument("--use-dashscope", action="store_true", help="启用DashScope生成摘要/多语翻译（需要API Key）")
    args = parser.parse_args()

    video_path = os.path.abspath(args.input)
    out_dir = args.outdir
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(video_path), "unified_output")
    out_dir = os.path.abspath(out_dir)
    _ensure_dir(out_dir)

    targets = [t.strip() for t in (args.targets or "").split(",") if t.strip()]
    report = build_report(
        video_path=video_path,
        out_dir=out_dir,
        targets=targets,
        whisper_model=args.whisper_model,
        max_keyframes=args.max_keyframes,
        min_gap=args.min_gap,
        with_srt=args.with_srt,
        use_dashscope=args.use_dashscope,
    )
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已生成：", report_path)
    print("关键帧目录：", os.path.join(out_dir, "keyframes"))
    if report.get("artifacts", {}).get("srt"):
        print("字幕：", report["artifacts"]["srt"])


if __name__ == "__main__":
    main()

