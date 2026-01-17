import argparse
import os
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from moviepy.video.fx.LumContrast import LumContrast
from moviepy.video.fx.GammaCorrection import GammaCorrection
import json

def detect_scenes(video_path, step=15, hist_thresh=0.45, motion_thresh=12.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev = None
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
            hist = cv2.calcHist([hsv], [0,1], None, [32,32], [0,180,0,256])
            hist = cv2.normalize(hist, None).flatten()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append((idx, float(np.mean(gray))))
            if prev is not None:
                d = cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)
                md = 0.0
                diff = cv2.absdiff(gray, prev_gray)
                md = float(np.mean(diff))
                motion_values.append((idx, md))
                if d > hist_thresh:
                    cuts.append(idx)
            prev = hist
            prev_gray = gray
        idx += 1
    cuts.append(total)
    cap.release()
    segments = []
    for i in range(len(cuts)-1):
        start = cuts[i]
        end = cuts[i+1]
        seg_motion = [m for f,m in motion_values if f>=start and f<end]
        avg_motion = float(np.mean(seg_motion)) if seg_motion else 0.0
        level = "high" if avg_motion >= motion_thresh else "low"
        seg_bright = [b for f,b in brightness_values if f>=start and f<end]
        avg_brightness = float(np.mean(seg_bright)) if seg_bright else 0.0
        segments.append({
            "start_frame": start,
            "end_frame": end,
            "avg_motion": avg_motion,
            "avg_brightness": avg_brightness,
            "level": level
        })
    return segments, fps

def apply_effects(clip, level, brightness):
    effects = []
    if level == "high":
        if brightness >= 180:
            effects.append(LumContrast(lum=-10, contrast=0.08, contrast_threshold=50))
            effects.append(GammaCorrection(gamma=0.98))
        elif brightness >= 130:
            effects.append(LumContrast(lum=-5, contrast=0.06, contrast_threshold=50))
        else:
            effects.append(GammaCorrection(gamma=1.06))
            effects.append(LumContrast(lum=0, contrast=0.05, contrast_threshold=50))
    else:
        if brightness >= 180:
            effects.append(LumContrast(lum=-8, contrast=0.05, contrast_threshold=50))
        elif brightness <= 110:
            effects.append(GammaCorrection(gamma=1.05))
        effects.append(FadeIn(duration=0.25))
        effects.append(FadeOut(duration=0.25))
    return clip.with_effects(effects), [
        {"effect": "LumContrast", "params": {"lum": -10 if brightness>=180 and level=="high" else (-8 if brightness>=180 and level=="low" else (0 if level=="high" and brightness<130 else -5 if level=="high" else None)), "contrast": 0.08 if brightness>=180 and level=="high" else (0.05 if level=="low" and brightness>=180 else (0.05 if level=="high" and brightness<130 else 0.06 if level=="high" and brightness>=130 else None)), "contrast_threshold": 50}},
        {"effect": "GammaCorrection", "params": {"gamma": 0.98 if brightness>=180 and level=="high" else (1.06 if level=="high" and brightness<130 else (1.05 if level=="low" and brightness<=110 else None))}},
        {"effect": "FadeIn", "params": {"duration": 0.25} if level=="low" else None},
        {"effect": "FadeOut", "params": {"duration": 0.25} if level=="low" else None},
    ]

def build_output(input_path, output_path, transition_fade=0.4, report_path=None):
    segments, fps = detect_scenes(input_path)
    base = VideoFileClip(input_path)
    clips = []
    report_segments = []
    for seg in segments:
        t_start = seg["start_frame"]/fps
        t_end = seg["end_frame"]/fps
        sub = base.subclipped(t_start, t_end)
        effected, effects_used = apply_effects(sub, seg["level"], seg.get("avg_brightness", 0.0))
        effected = effected.with_effects([FadeIn(duration=transition_fade), FadeOut(duration=transition_fade)])
        clips.append(effected)
        report_segments.append({
            "start_time": t_start,
            "end_time": t_end,
            "level": seg["level"],
            "avg_motion": seg["avg_motion"],
            "avg_brightness": seg.get("avg_brightness", 0.0),
            "effects": [e for e in effects_used if e.get("params") is not None]
        })
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    base.close()
    final.close()
    if report_path:
        report = {
            "input": input_path,
            "output": output_path,
            "segments": report_segments
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return segments

def main():
    parser = argparse.ArgumentParser(description="特效自动添加")
    parser.add_argument("--input", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output", type=str, default="output_with_effects.mp4", help="输出视频路径")
    parser.add_argument("--transition-fade", type=float, default=0.4, help="转场淡入淡出时长")
    parser.add_argument("--report", type=str, default=None, help="输出特效应用的JSON报告路径")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report_path = args.report or (os.path.splitext(args.output)[0] + "_effects_report.json")
    segs = build_output(args.input, args.output, args.transition_fade, report_path)
    print("Segments:", segs)

if __name__ == "__main__":
    main()
