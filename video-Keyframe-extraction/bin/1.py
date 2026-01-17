import os
import subprocess
from PIL import Image, ImageDraw
import imagehash
import re
import random
import string

# ================== 可配置参数 ==================
video_folder = r"D:\rubbish\video-handle\video-Keyframe-extraction\bin"
output_base = os.path.join(video_folder, "frames_output")
ffmpeg_path = r"D:\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

# 忽略区域（右下角时间戳）
IGNORE_HEIGHT_RATIO = 0.20       # 忽略底部 20% 高度
IGNORE_WIDTH_RATIO = 0.30        # 忽略右侧 30% 宽度

# 去重阈值
DIFF_THRESHOLD = 6

# OCR 友好的 drawtext 样式
DRAWTEXT = (
    "select='eq(pict_type\\,I)',"
    "drawtext=text='%{pts\\:hms}':"
    "x=w-tw-50:y=h-th-50:"
    "fontsize=72:fontcolor=white:"
    "box=1:boxcolor=black@0.8:boxborderw=10:"
    "font='Arial'"
)

# 支持的视频后缀
video_extensions = (".mp4", ".mov", ".mkv", ".avi")
# =================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ================== 清理文件名 ==================
def sanitize_filenames(folder: str, exts=(".mp4", ".mov", ".mkv", ".avi")):
    for fname in os.listdir(folder):
        if not fname.lower().endswith(exts):
            continue
        old_path = os.path.join(folder, fname)
        name, ext = os.path.splitext(fname)

        # 删除中文
        safe_name = re.sub(r"[\u4e00-\u9fff]", "", name)
        # 非法字符替换成下划线
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", safe_name)
        # 去掉连续下划线和首尾下划线/空格
        safe_name = re.sub(r"_+", "_", safe_name).strip("_ ")

        if not safe_name:
            safe_name = "video"

        new_name = safe_name + ext
        new_path = os.path.join(folder, new_name)

        # 避免重名
        counter = 1
        while os.path.exists(new_path):
            new_name = f"{safe_name}_{counter}{ext}"
            new_path = os.path.join(folder, new_name)
            counter += 1

        if new_path != old_path:
            os.rename(old_path, new_path)
            print(f"[重命名] {fname} -> {new_name}")
# ============================================================

def run_ffmpeg_extract(video_path: str, output_folder: str):
    """
    提取关键帧并在右下角叠加时间戳。
    """
    ensure_dir(output_folder)
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", video_path,
        "-filter_complex", DRAWTEXT,
        "-vsync", "vfr",
        os.path.join(output_folder, "frame_%05d.jpg")
    ]
    subprocess.run(cmd, check=True)

def hash_ignoring_stamp(img: Image.Image,
                        ignore_width_ratio: float = IGNORE_WIDTH_RATIO,
                        ignore_height_ratio: float = IGNORE_HEIGHT_RATIO):
    """
    生成感知哈希，忽略右下角时间戳区域。
    """
    w, h = img.size
    x0 = int(w * (1 - ignore_width_ratio))  # 右边部分
    x1 = w
    y0 = int(h * (1 - ignore_height_ratio))  # 底部部分
    y1 = h

    masked = img.convert("RGB").copy()
    draw = ImageDraw.Draw(masked)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    return imagehash.phash(masked)

def dedupe_folder(folder: str,
                  ignore_width_ratio: float = IGNORE_WIDTH_RATIO,
                  ignore_height_ratio: float = IGNORE_HEIGHT_RATIO,
                  diff_threshold: int = DIFF_THRESHOLD) -> int:
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()
    seen_hashes = []
    removed = 0

    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            with Image.open(fpath) as img:
                h = hash_ignoring_stamp(img, ignore_width_ratio, ignore_height_ratio)
        except Exception as e:
            print(f"[跳过] 打不开图片: {fpath} ({e})")
            continue

        if any(h - sh < diff_threshold for sh in seen_hashes):
            os.remove(fpath)
            print(f"[删除] {fname} (重复帧)")
            removed += 1
        else:
            seen_hashes.append(h)

    return removed

# ================== 随机后缀重命名 ==================
def rename_with_random_suffix(folder: str, exts=(".jpg", ".jpeg", ".png"), suffix_len=5):
    """
    给图片文件名加随机后缀 (字母+数字)，保留原编号
    """
    for fname in os.listdir(folder):
        if not fname.lower().endswith(exts):
            continue
        old_path = os.path.join(folder, fname)
        name, ext = os.path.splitext(fname)

        rand_str = "".join(random.choices(string.ascii_letters + string.digits, k=suffix_len))
        new_name = f"{name}_{rand_str}{ext}"
        new_path = os.path.join(folder, new_name)

        # 避免重名
        while os.path.exists(new_path):
            rand_str = "".join(random.choices(string.ascii_letters + string.digits, k=suffix_len))
            new_name = f"{name}_{rand_str}{ext}"
            new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        print(f"[随机重命名] {fname} -> {new_name}")
# =====================================================

def main():
    ensure_dir(output_base)

    print("[0/3] 清理文件名 ...")
    sanitize_filenames(video_folder)

    for video_file in os.listdir(video_folder):
        if not video_file.lower().endswith(video_extensions):
            continue
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        outdir = os.path.join(output_base, video_name)
        ensure_dir(outdir)

        print(f"[1/3] 提取关键帧并添加时间戳 => {video_file}")
        run_ffmpeg_extract(video_path, outdir)

        print(f"[2/3] 去重（忽略右下角时间戳）...")
        removed = dedupe_folder(outdir)
        kept = len([f for f in os.listdir(outdir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"完成：{video_file}，删除重复 {removed} 张，保留 {kept} 张。")

        print(f"[3/3] 开始随机重命名 ...")
        rename_with_random_suffix(outdir)

    print("全部处理完成！")


if __name__ == "__main__":
    main()
