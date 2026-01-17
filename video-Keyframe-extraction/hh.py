import os
import sys
import pyperclip
import tkinter as tk

# ------- 配置（如需修改路径，直接改这里） -------
folder_a = r"C:\Users\toms\Downloads\ffmpeg-7.1.1-essentials_build\bin\frames_output\ocr_results"
folder_b = r"C:\Users\toms\Videos\pyvideotrans\recogn"
MAX_KB = 18  # 每块最大 KB
# --------------------------------------------------

def find_files_by_number(folder, number):
    number = str(number)
    if not os.path.isdir(folder):
        return []
    matches = []
    for fname in sorted(os.listdir(folder)):  # 排序方便按顺序读
        if number in fname:
            matches.append(os.path.join(folder, fname))
    return matches

def read_text_file(path):
    encs = ("utf-8", "utf-8-sig", "gbk", "latin1")
    for e in encs:
        try:
            with open(path, "r", encoding=e) as f:
                return f.read()
        except Exception:
            continue
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def copy_to_clipboard(text):
    try:
        pyperclip.copy(text)
        return True, "pyperclip"
    except Exception:
        try:
            r = tk.Tk()
            r.withdraw()
            r.clipboard_clear()
            r.clipboard_append(text)
            r.update()
            r.destroy()
            return True, "tkinter"
        except Exception as e:
            return False, str(e)

def merge_srt_text(s):
    """针对 SRT 文件：删除换行，保留行号和时间戳"""
    lines = s.strip().splitlines()
    merged = [line.strip() for line in lines if line.strip()]
    return " ".join(merged)

def split_text_by_kb(text, max_kb=MAX_KB):
    """
    将 text 按 max_kb KB 拆分，返回列表
    尽量按句号或空格拆分，避免中文半截
    """
    encoded = text.encode('utf-8')
    total_bytes = len(encoded)
    if total_bytes <= max_kb * 1024:
        return [text]

    # 粗略按字数分块
    num_blocks = (total_bytes // (max_kb * 1024)) + 1
    approx_chars = len(text) // num_blocks
    blocks = []
    start = 0
    while start < len(text):
        end = start + approx_chars
        if end < len(text):
            # 尝试往后找到句号或空格，避免半句
            for i in range(end, min(end + 50, len(text))):
                if text[i] in "。！？.!? \n":
                    end = i + 1
                    break
        blocks.append(text[start:end].strip())
        start = end
    return blocks

def main():
    number = input("请输入数字（例如 6）: ").strip()
    if not number:
        print("你没有输入数字，程序退出。")
        return

    a_files = find_files_by_number(folder_a, number)
    b_files = find_files_by_number(folder_b, number)
    all_files = a_files + b_files

    if not all_files:
        print(f"在两个目录中都没找到包含“{number}”的文件。")
        return

    for idx, p in enumerate(all_files, start=1):
        basename = os.path.basename(p)
        txt = read_text_file(p).strip()
        if not txt:
            print(f"[跳过] {basename} 内容为空")
            continue

        txt = merge_srt_text(txt)
        blocks = split_text_by_kb(txt, MAX_KB)

        for b_idx, block in enumerate(blocks, start=1):
            ok, info = copy_to_clipboard(block)
            if ok:
                print(f"({idx}/{len(all_files)}) 已复制 {basename} 第 {b_idx}/{len(blocks)} 块到剪贴板（{info}）")
            else:
                print(f"复制失败：{info}。你可以手动打开文件：{p}")
                break

            if b_idx < len(blocks):
                input("按 Enter 复制下一块内容...")

        if idx < len(all_files):
            input("按 Enter 复制下一个文件内容...")
        else:
            print("✅ 所有文件都已复制完毕。")

if __name__ == "__main__":
    main()
