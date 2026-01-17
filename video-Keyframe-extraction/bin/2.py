import os
from paddleocr import PaddleOCR
from PIL import Image

# 设置 UTF-8 编码以支持中文路径
os.environ["PYTHONIOENCODING"] = "utf-8"

# ========== 配置 ==========
frames_output = r"D:\rubbish\video-handle\video-Keyframe-extraction\bin\frames_output"
output_dir = os.path.join(frames_output, "ocr_results")

# 初始化 PaddleOCR
ocr = PaddleOCR(use_textline_orientation=True, lang="ch")

# ========== 初始化 ==========
os.makedirs(output_dir, exist_ok=True)

# ========== 遍历文件夹 ==========
for folder_name in os.listdir(frames_output):
    folder_path = os.path.normpath(os.path.join(frames_output, folder_name))
    if not os.path.isdir(folder_path):
        print(f"跳过非文件夹: {folder_path}")
        continue

    output_file = os.path.join(output_dir, f"{folder_name}.txt")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                file_path = os.path.normpath(os.path.join(folder_path, file_name))
                try:
                    # 验证图像是否可打开
                    with Image.open(file_path) as img:
                        img.verify()
                    with Image.open(file_path):
                        pass

                    # OCR 处理
                    result = ocr.predict(file_path)
                    if result:
                        all_texts = []
                        for res in result:
                            rec_texts = res.get("rec_texts", [])
                            rec_scores = res.get("rec_scores", [])
                            boxes = res.get("rec_polys", [])

                            for t, s, box in zip(rec_texts, rec_scores, boxes):
                                if not t.strip():
                                    continue

                                # ---------- 四舍五入到小数点后一位 ----------
                                try:
                                    sval = float(s)
                                    rounded = round(sval + 1e-12, 1)
                                    if rounded >= 1.0:
                                        score_str = "1"
                                    else:
                                        score_str = f"{rounded:.1f}"
                                except:
                                    score_str = "0"

                                # ---------- 坐标处理（只取前两个点） ----------
                                try:
                                    point1 = f"{int(box[0][0])},{int(box[0][1])}"
                                    point2 = f"{int(box[1][0])},{int(box[1][1])}"
                                    box_str = f"{point1} {point2}"
                                except:
                                    box_str = ""

                                all_texts.append(f"{t}({score_str}){box_str}")

                        text = "\n".join(all_texts).strip()
                        if text:
                            f_out.write(f"【{file_name}】\n{text}\n\n")
                        else:
                            print(f"无文本: {file_path}")
                    else:
                        print(f"无结果: {file_path}")
                except Exception as e:
                    print(f"出错: {file_path} | {str(e)}")
                    continue

print(f"OCR 完成 ✅\n结果保存在: {output_dir}")
input("按 Enter 键结束...")
