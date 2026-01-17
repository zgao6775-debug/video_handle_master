import hashlib
import json
import os
import subprocess


def _run(cmd, cwd=None, log=None):
    if log:
        log(" ".join(cmd))
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
    if proc.stdout and log:
        for line in proc.stdout:
            log(line.rstrip("\n"))
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"命令执行失败，退出码 {code}")


def _list_audio_files(root_dir):
    exts = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
    out = []
    for cur, _dirs, files in os.walk(root_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                out.append(os.path.join(cur, name))
    out.sort()
    return out


def _sample_visual_features(video_path, step_sec=1.0, max_samples=60):
    try:
        import cv2
        import numpy as np
    except Exception:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * float(step_sec))))

    sampled = 0
    idx = 0
    prev_gray = None

    brightness_vals = []
    saturation_vals = []
    motion_vals = []

    while sampled < max_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(float(np.mean(gray)))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_vals.append(float(np.mean(hsv[:, :, 1])))

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_vals.append(float(np.mean(diff)))
        prev_gray = gray

        sampled += 1
        idx += step

    cap.release()

    def avg(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "brightness_mean": avg(brightness_vals),
        "saturation_mean": avg(saturation_vals),
        "motion_mean": avg(motion_vals),
        "visual_samples": int(sampled),
    }


def _sample_audio_features(video_path, max_seconds=90, sample_rate=22050):
    try:
        import numpy as np
        from moviepy.editor import VideoFileClip
    except Exception:
        return {}

    clip = VideoFileClip(video_path)
    try:
        if clip.audio is None:
            return {"has_audio": False}
        t_end = min(float(max_seconds), float(clip.duration or 0))
        if t_end <= 0:
            return {"has_audio": True, "audio_rms": 0.0, "audio_zcr": 0.0}
        sub = clip.subclip(0, t_end)
        arr = sub.audio.to_soundarray(fps=int(sample_rate))
        if arr is None or len(arr) == 0:
            return {"has_audio": True, "audio_rms": 0.0, "audio_zcr": 0.0}
        mono = arr.mean(axis=1)
        rms = float(np.sqrt(np.mean(np.square(mono))))
        zcr = float(np.mean(np.abs(np.diff((mono > 0).astype(np.int8)))))
        return {"has_audio": True, "audio_rms": rms, "audio_zcr": zcr, "audio_seconds": float(t_end)}
    finally:
        try:
            clip.close()
        except Exception:
            pass


def _load_report_text(report_path):
    if not report_path or not os.path.exists(report_path):
        return None
    try:
        data = json.loads(open(report_path, "r", encoding="utf-8", errors="ignore").read())
    except Exception:
        return None
    transcript = (data.get("transcript") or {})
    summaries = (data.get("summaries") or {})
    lang = (transcript.get("language") or "").strip() or "unknown"
    transcript_text = (transcript.get("text") or "").strip()
    summary_text = ""
    if isinstance(summaries, dict):
        summary_text = (summaries.get(lang) or summaries.get("zh") or summaries.get("en") or "").strip()
    parts = [p for p in [summary_text, transcript_text] if p]
    if not parts:
        return None
    return {"language": lang, "text": "\n".join(parts), "report_path": os.path.abspath(report_path)}


def _load_srt_text(srt_path):
    if not srt_path or not os.path.exists(srt_path):
        return None
    try:
        raw = open(srt_path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    except Exception:
        return None
    lines = []
    for line in raw:
        s = line.strip()
        if not s:
            continue
        if s.isdigit():
            continue
        if "-->" in s:
            continue
        lines.append(s)
    text = " ".join(lines).strip()
    if not text:
        return None
    return {"language": "unknown", "text": text, "srt_path": os.path.abspath(srt_path)}


def _semantic_scores(text):
    if not text:
        return None
    t = text.lower()

    groups = {
        "happy": [
            "happy",
            "joy",
            "fun",
            "smile",
            "laugh",
            "celebrat",
            "party",
            "love",
            "excited",
            "good",
            "nice",
            "awesome",
            "cool",
            "great",
            "wonderful",
            "amazing",
            "开心",
            "高兴",
            "快乐",
            "欢快",
            "喜悦",
            "庆祝",
            "惊喜",
            "喜欢",
            "爱",
            "笑",
            "有趣",
        ],
        "sad": [
            "sad",
            "cry",
            "miss",
            "lonely",
            "regret",
            "sorry",
            "lost",
            "pain",
            "depress",
            "tired",
            "bad",
            "terrible",
            "awful",
            "伤心",
            "难过",
            "悲伤",
            "失落",
            "遗憾",
            "抱歉",
            "孤独",
            "想念",
            "哭",
            "痛苦",
            "疲惫",
        ],
        "tense": [
            "tense",
            "scary",
            "fear",
            "panic",
            "danger",
            "alert",
            "chase",
            "fight",
            "warning",
            "mystery",
            "suspense",
            "紧张",
            "害怕",
            "恐怖",
            "危险",
            "追",
            "打",
            "战斗",
            "警告",
            "悬疑",
            "惊悚",
        ],
        "energetic": [
            "run",
            "dance",
            "sport",
            "game",
            "fast",
            "go",
            "let's",
            "yeah",
            "wow",
            "power",
            "hype",
            "激动",
            "燃",
            "加油",
            "冲",
            "奔跑",
            "运动",
            "跳舞",
            "比赛",
            "速度",
            "热血",
        ],
        "calm": [
            "calm",
            "quiet",
            "relax",
            "peace",
            "soft",
            "slow",
            "gentle",
            "sleep",
            "nature",
            "breeze",
            "平静",
            "安静",
            "放松",
            "舒缓",
            "治愈",
            "温柔",
            "慢",
            "自然",
            "微风",
            "睡",
        ],
    }

    def count_terms(terms):
        score = 0.0
        for w in terms:
            if w.isascii():
                if w in t:
                    score += 1.0
            else:
                if w in text:
                    score += 1.0
        return score

    scores = {k: count_terms(v) for k, v in groups.items()}

    bangs = text.count("!") + text.count("！")
    questions = text.count("?") + text.count("？")
    scores["energetic"] += bangs * 0.25
    scores["tense"] += questions * 0.10

    if max(scores.values() or [0.0]) <= 0:
        return None

    return scores


def infer_mood(video_path, text_context=None):
    v = _sample_visual_features(video_path)
    a = _sample_audio_features(video_path)

    brightness = float(v.get("brightness_mean", 0.0))
    saturation = float(v.get("saturation_mean", 0.0))
    motion = float(v.get("motion_mean", 0.0))
    rms = float(a.get("audio_rms", 0.0))
    zcr = float(a.get("audio_zcr", 0.0))

    energetic_score = motion * 0.08 + rms * 8.0 + zcr * 2.0
    calm_score = (40.0 - motion) * 0.08 + (0.25 - rms) * 6.0
    sad_score = (110.0 - brightness) * 0.04 + (0.25 - rms) * 5.0
    happy_score = (brightness - 90.0) * 0.04 + (saturation - 70.0) * 0.02 + (rms - 0.12) * 3.0
    tense_score = (motion - 18.0) * 0.06 + (110.0 - brightness) * 0.03 + zcr * 1.5

    scores_stats = {
        "energetic": energetic_score,
        "calm": calm_score,
        "sad": sad_score,
        "happy": happy_score,
        "tense": tense_score,
    }

    scores_semantic = None
    text_used = None
    if isinstance(text_context, dict):
        text_used = (text_context.get("text") or "").strip()
    elif isinstance(text_context, str):
        text_used = text_context.strip()
    if text_used:
        scores_semantic = _semantic_scores(text_used)

    scores = dict(scores_stats)
    if scores_semantic:
        semantic_weight = 2.0 if len(text_used) >= 60 else 1.2
        for k, v_sem in scores_semantic.items():
            scores[k] = float(scores.get(k, 0.0)) + float(v_sem) * semantic_weight

    mood = max(scores.items(), key=lambda kv: kv[1])[0]
    feats = {}
    feats.update(v)
    feats.update(a)
    feats["scores_stats"] = scores_stats
    if scores_semantic:
        feats["scores_semantic"] = scores_semantic
        feats["text_len"] = len(text_used)
        if isinstance(text_context, dict):
            feats["text_language"] = text_context.get("language")
            feats["report_path"] = text_context.get("report_path")
    feats["scores_combined"] = scores
    return mood, feats


def select_bgm(library_dir, mood, video_path=None):
    if not library_dir or not os.path.isdir(library_dir):
        raise RuntimeError("请设置有效的本地曲库目录")

    mapping = {
        "calm": ["calm", "平静", "舒缓", "治愈"],
        "happy": ["happy", "欢快", "轻快", "明亮"],
        "sad": ["sad", "伤感", "忧郁", "低沉"],
        "tense": ["tense", "紧张", "悬疑", "压迫"],
        "energetic": ["energetic", "激昂", "燃", "动感", "节奏"],
    }

    wanted = mapping.get(mood, [mood])
    candidates = []
    for key in wanted:
        d = os.path.join(library_dir, key)
        if os.path.isdir(d):
            candidates.extend(_list_audio_files(d))

    if not candidates:
        all_files = _list_audio_files(library_dir)
        if not all_files:
            raise RuntimeError("曲库目录中未找到音频文件（wav/mp3/m4a/aac/flac/ogg）")
        candidates = all_files

    seed = (video_path or "") + "|" + mood
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    idx = int(h[:8], 16) % len(candidates)
    return candidates[idx]


def combine_bgm(video_path, bgm_path, output_path, volume=0.7, log=None):
    if not os.path.exists(video_path):
        raise RuntimeError("未找到视频文件")
    if not os.path.exists(bgm_path):
        raise RuntimeError("未找到背景音乐文件")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vol = float(volume)
    if vol <= 0:
        raise RuntimeError("音量系数必须大于 0")

    filter_complex = f"[1:a]volume={vol}[bgm]"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        bgm_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        "[bgm]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    _run(cmd, log=log)
    return output_path


def smart_bgm(video_path, out_dir, library_dir, volume=0.7, log=None, unified_dir=None, report_path=None, text_context=None):
    os.makedirs(out_dir, exist_ok=True)
    report_text = None
    if report_path:
        report_text = _load_report_text(report_path)
    if not report_text and unified_dir:
        report_text = _load_report_text(os.path.join(unified_dir, "report.json"))
    if not report_text and unified_dir:
        subs_dir = os.path.join(unified_dir, "subtitles")
        if os.path.isdir(subs_dir):
            try:
                srt_files = [os.path.join(subs_dir, n) for n in os.listdir(subs_dir) if n.lower().endswith(".srt")]
            except Exception:
                srt_files = []
            srt_files.sort()
            if srt_files:
                report_text = _load_srt_text(srt_files[0])
    if not report_text and isinstance(text_context, str) and text_context.strip():
        report_text = {"language": "unknown", "text": text_context.strip()}
    mood, feats = infer_mood(video_path, text_context=report_text)
    bgm = select_bgm(library_dir, mood, video_path=video_path)
    out_video = os.path.join(out_dir, "video_with_music.mp4")
    out_meta = os.path.join(out_dir, "bgm_meta.json")
    combine_bgm(video_path, bgm, out_video, volume=volume, log=log)
    meta = {
        "mode": "smart_bgm",
        "mood": mood,
        "bgm_path": os.path.abspath(bgm),
        "volume": float(volume),
        "features": feats,
    }
    if report_text:
        meta["text_context"] = {
            "language": report_text.get("language"),
            "text": report_text.get("text"),
            "report_path": report_text.get("report_path"),
        }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_video, out_meta

