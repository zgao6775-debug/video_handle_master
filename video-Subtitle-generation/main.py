import os
import subprocess
import whisper
import torch

# --- 1. 配置文件路径 ---
# 请确保你的视频文件路径正确
video_file_path = r"D:/rubbish/video-handle/video-Subtitle-generation/video/test1.mp4"
# 生成的SRT字幕文件路径
srt_file_path = r"D:/rubbish/video-handle/video-Subtitle-generation/output/output.srt"
# 最终带字幕的视频输出路径
output_video_path = r"D:/rubbish/video-handle/video-Subtitle-generation/output/output_video.mp4"


# --- 2. GPU检测 ---
def check_gpu():
    """检测GPU是否可用"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ 检测到GPU: {gpu_name}")
        print(f"   - CUDA版本: {torch.version.cuda}")
        print(f"   - 显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        return "cuda"
    else:
        print("⚠️  未检测到CUDA，将使用CPU（速度较慢）")
        print("   提示: 如需GPU加速，请安装CUDA版本的PyTorch")
        print(
            "   安装命令: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return "cpu"


# --- 3. 核心功能函数 ---
def transcribe_video_to_srt(video_path, srt_path, model_name="medium", device="cuda"):
    """
    使用Whisper模型转录视频并生成带精确时间轴的SRT字幕文件（GPU加速版）

    参数:
        video_path: 视频文件路径
        srt_path: 输出SRT字幕路径
        model_name: Whisper模型大小 (tiny/base/small/medium/large)
        device: 运行设备 ("cuda" 或 "cpu")
    """
    print(f"\n1. 正在加载Whisper模型 '{model_name}' 到 {device.upper()}...")
    model = whisper.load_model(model_name, device=device)

    print(f"2. 正在转录视频: {os.path.basename(video_path)}")
    # 使用GPU加速转录，fp16可以提升速度（仅GPU支持）
    result = model.transcribe(
        video_path,
        language="zh",  # 使用"zh"而非"Chinese"（标准ISO代码）
        fp16=(device == "cuda"),  # GPU时使用半精度加速
        verbose=True  # 显示进度
    )

    print(f"3. 正在生成SRT字幕文件: {os.path.basename(srt_path)}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(srt_path), exist_ok=True)

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"]):
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()

            # 格式化时间为SRT格式
            start_srt = format_time(start_time)
            end_srt = format_time(end_time)

            # 写入SRT条目
            f.write(f"{i + 1}\n")
            f.write(f"{start_srt} --> {end_srt}\n")
            f.write(f"{text}\n\n")

    print(f"✅ SRT字幕文件已成功生成: {srt_path}")


def format_time(seconds):
    """将秒数格式化为SRT时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"


def embed_subtitles_into_video(video_path, srt_path, output_path):
    """
    使用ffmpeg将SRT字幕烧录到视频（硬字幕）
    改进版：针对Windows路径优化
    """
    print(f"\n4. 正在将字幕烧录到视频（硬字幕）...")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Windows路径处理：使用双反斜杠，然后转义冒号
    srt_path_abs = os.path.abspath(srt_path)
    # 方法1：使用正斜杠并转义冒号（适用于新版ffmpeg）
    srt_path_fixed = srt_path_abs.replace("\\", "/").replace(":", "\\:")

    # 构建字幕滤镜（使用单引号包裹路径）
    vf_param = f"subtitles='{srt_path_fixed}'"

    # 构建ffmpeg命令
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", vf_param,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-c:a", "copy",
        "-y",
        output_path
    ]

    print(f"   字幕文件路径: {srt_path_abs}")
    print(f"   转换后路径: {srt_path_fixed}")
    print(f"   执行命令: ffmpeg -i [视频] -vf \"{vf_param}\" [输出]\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        print(f"✅ 带字幕的视频已成功生成: {output_path}")
        print(f"   字幕已永久烧录到视频中，可在任何播放器直接观看")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 方案1失败，尝试备用方案...")
        print(f"   错误信息: {e.output[-500:]}")  # 只显示最后500字符
        # 尝试备用方案
        return embed_subtitles_fallback(video_path, srt_path, output_path)
    except FileNotFoundError:
        print("❌ 未找到ffmpeg！请先安装ffmpeg")
        print("   下载: https://www.gyan.dev/ffmpeg/builds/")
        raise


def embed_subtitles_fallback(video_path, srt_path, output_path):
    """
    备用方案1：使用双反斜杠路径
    """
    print(f"\n   尝试备用方案1（双反斜杠路径）...")

    # 使用双反斜杠
    srt_path_escaped = os.path.abspath(srt_path).replace("\\", "\\\\\\\\")
    vf_param = f"subtitles={srt_path_escaped}"

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", vf_param,
        "-c:v", "libx264",
        "-crf", "23",
        "-c:a", "copy",
        "-y",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ 备用方案1成功！视频已生成: {output_path}")
        return True
    except Exception:
        print(f"   备用方案1失败，尝试方案2...")
        return embed_subtitles_fallback2(video_path, srt_path, output_path)


def embed_subtitles_fallback2(video_path, srt_path, output_path):
    """
    备用方案2：先转换SRT为ASS，使用ass滤镜
    """
    print(f"\n   尝试备用方案2（ASS格式）...")

    # 转换SRT为ASS格式
    ass_path = srt_path.replace(".srt", ".ass")
    convert_srt_to_ass(srt_path, ass_path)

    # 使用正斜杠路径
    ass_path_fixed = os.path.abspath(ass_path).replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"ass='{ass_path_fixed}'",
        "-c:v", "libx264",
        "-crf", "23",
        "-c:a", "copy",
        "-y",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ 备用方案2成功！视频已生成: {output_path}")
        return True
    except Exception:
        print(f"   备用方案2失败，尝试方案3...")
        return embed_subtitles_fallback3(video_path, srt_path, output_path)


def embed_subtitles_fallback3(video_path, srt_path, output_path):
    """
    备用方案3：使用filename参数（最通用）
    """
    print(f"\n   尝试备用方案3（使用filename参数）...")

    # 使用正斜杠（最通用的方式）
    srt_path_simple = os.path.abspath(srt_path).replace("\\", "/")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles=filename='{srt_path_simple}'",
        "-c:v", "libx264",
        "-crf", "23",
        "-c:a", "copy",
        "-y",
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        print(f"✅ 备用方案3成功！视频已生成: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 所有方案都失败了！")
        print(f"\n完整错误信息:")
        print(e.output)
        print(f"\n请手动测试以下命令:")
        print(
            f"ffmpeg -i \"{video_path}\" -vf \"subtitles='{srt_path_simple}'\" -c:v libx264 -c:a copy \"{output_path}\"")
        return False


def convert_srt_to_ass(srt_path, ass_path):
    """简单的SRT转ASS转换"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    # ASS文件头
    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_header)

        # 解析SRT并转换为ASS
        blocks = srt_content.strip().split('\n\n')
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                time_line = lines[1]
                text = ' '.join(lines[2:])

                # 转换时间格式
                times = time_line.split(' --> ')
                if len(times) == 2:
                    start = times[0].replace(',', '.').strip()
                    end = times[1].replace(',', '.').strip()
                    f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")


def embed_subtitles_soft(video_path, srt_path, output_path):
    """
    备用方案：生成软字幕（可在播放器中开关）
    如果硬字幕嵌入失败，可以尝试这个方法
    """
    print(f"\n4. 正在将字幕添加为软字幕（可在播放器中开关）...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", srt_path,
        "-c:v", "copy",  # 视频直接复制（速度快）
        "-c:a", "copy",  # 音频直接复制
        "-c:s", "mov_text",  # 字幕编码（MP4格式）
        "-metadata:s:s:0", "language=chi",  # 标记为中文字幕
        "-y",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ 带软字幕的视频已成功生成: {output_path}")
        print("   注意: 需要在支持字幕的播放器中手动开启字幕显示")
    except subprocess.CalledProcessError as e:
        print(f"❌ 添加软字幕失败: {e.stderr}")
        raise


# --- 4. 主程序入口 ---
if __name__ == "__main__":
    print("=" * 60)
    print("   🎬 视频字幕自动生成工具 (GPU加速版)")
    print("=" * 60)

    # 检测GPU
    device = check_gpu()

    # 检查视频文件是否存在
    if not os.path.exists(video_file_path):
        print(f"\n❌ 错误: 视频文件不存在于 '{video_file_path}'")
        print("   请检查路径是否正确")
    else:
        try:
            # 步骤一：转录视频并生成SRT字幕（使用GPU加速）
            transcribe_video_to_srt(
                video_file_path,
                srt_file_path,
                model_name="medium",  # 可改为 small/large 调整精度
                device=device
            )

            # 步骤二：将生成的SRT字幕烧录到视频（硬字幕，任何播放器可见）
            embed_subtitles_into_video(video_file_path, srt_file_path, output_video_path)

            print("\n" + "=" * 60)
            print("🎉 所有任务已成功完成！")
            print("=" * 60)
            print(f"📁 输出文件:")
            print(f"   - 字幕文件: {srt_file_path}")
            print(f"   - 视频文件（硬字幕）: {output_video_path}")
            print(f"\n💡 字幕已永久烧录到视频，可在任何播放器直接观看！")

        except Exception as e:
            print(f"\n❌ 程序执行失败: {str(e)}")
            print("\n请检查:")
            print("1. ffmpeg是否正确安装: ffmpeg -version")
            print("2. 视频文件路径是否正确")
            print("3. SRT字幕文件是否成功生成")