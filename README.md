# 智能视频自动化剪辑系统 (Intelligent Video Auto-Editing System)

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

这是一个基于多模态AI技术构建的智能视频自动化剪辑系统。它能够将无序的原始视频素材，自动处理成具备叙事逻辑、背景音乐和基础特效的精剪短片，旨在将视频创作者从重复性的初剪工作中解放出来。

## ✨ 主要功能

- **自动镜头切分 (Automatic Shot Segmentation):** 基于颜色直方图的Bhattacharyya距离，快速准确地将长视频分割成独立的镜头单元。
- **多模态内容理解 (Multi-modal Content Analysis):**
    - **目标检测:** 利用YOLOv3识别画面中的常见物体。
    - **动作分析:** 基于MediaPipe Pose捕捉人物姿态与基本动作。
    - **语音识别:** 使用OpenAI Whisper将视频中的语音精准转换为文本字幕。
- **LLM驱动的叙事剪辑 (LLM-Driven Narrative Editing):** 创新性地利用大语言模型（阿里通义千问）的推理能力，根据视频内容的结构化描述，自动生成符合逻辑和叙事性的剪辑序列。
- **智能配乐 (Smart BGM Selection):** 融合视觉、听觉和文本特征，综合评估视频的整体情感基调，从本地曲库中自动匹配最合适的背景音乐。
- **自动化视觉效果 (Automatic Visual Effects):** 根据镜头的亮度、运动幅度等属性，自动应用亮度/对比度校正、Gamma校正、淡入淡出等基础特效。
- **图形化用户界面 (GUI):** 提供简洁直观的Tkinter界面，支持一键式操作，轻松启动整个自动化处理流程。

## 🏛️ 系统架构

项目采用分层解耦的模块化架构，确保了系统的可扩展性和可维护性。

1.  **基础算子层 (Infrastructure Layer):** 封装FFmpeg、OpenCV等底层工具，提供标准化的原子操作接口（如视频读写、拼接、混音）。
2.  **核心算法层 (Algorithm Layer):** 集成YOLOv3、MediaPipe、Whisper等专用AI模型，负责执行具体的感知与分析任务。
3.  **业务编排层 (Orchestration Layer):** 系统的“指挥官”，负责定义处理流水线，管理模块间的数据流，并调用大语言模型进行高级决策。

## 🛠️ 技术栈

- **编程语言:** Python 3.9
- **核心框架与库:**
    - **媒体处理:** OpenCV, FFmpeg, MoviePy
    - **AI / 深度学习:** PyTorch, MediaPipe, OpenAI-Whisper
    - **大语言模型:** Alibaba DashScope (Qwen)
    - **GUI:** Tkinter

## 🚀 安装与配置

请按照以下步骤在您的本地环境中配置和运行本项目。

### 1. 先决条件

- **Python:** 确保您已安装 Python 3.9 或更高版本。
- **FFmpeg:** 这是一个核心依赖，请务必预先安装，并将其可执行文件路径添加到系统的环境变量（`PATH`）中。
    - [FFmpeg 官方下载地址](https://ffmpeg.org/download.html)

### 2. 克隆仓库

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 3. 创建虚拟环境并安装依赖

建议使用虚拟环境以隔离项目依赖。

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 安装依赖库
pip install -r requirements.txt
```
*(注意: 您需要先根据项目实际使用的库生成 `requirements.txt` 文件，可以使用 `pip freeze > requirements.txt` 命令)*

### 4. 下载预训练模型

- **YOLOv3 模型:** 请从YOLO官网下载 `yolov3.weights` 和 `yolov3.cfg` 文件，并将它们放置在 `video-objects-detect/yolo` 目录下。
    - `yolov3.weights` 下载链接: [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

### 5. 配置API密钥

本项目使用阿里云DashScope调用大语言模型。请在 `config.py` (如果不存在请自行创建) 文件中配置您的API密钥。

```python
# config.py
DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## ▶️ 如何使用

1.  **启动GUI程序:**
    ```bash
    python GUI_code/app.py
    ```
2.  **选择视频:** 在弹出的图形界面中，点击“选择视频”按钮，找到您想要处理的原始视频文件。
3.  **开始处理:** 点击“开始处理”按钮，系统将启动全自动化的剪辑流水线。您可以在下方的日志窗口中实时查看处理进度。
4.  **查看结果:** 处理完成后，所有输出文件（包括最终视频、中间数据等）将保存在项目根目录下的 `output/{RunID}` 文件夹中。

## 📂 项目结构

```
.
├── GUI_code/                 # GUI界面相关代码
├── Video-auto-editing/       # 自动剪辑与视频合成模块
├── video-effects-add/        # 自动视觉效果模块
├── video-Motion-Analysis/    # 动作分析模块 (MediaPipe)
├── video-objects-detect/     # 目标检测模块 (YOLO)
├── video-unified-output/     # 统一输出与镜头切分模块
├── Video_BGM/                # 智能配乐模块
├── output/                   # 所有输出结果的存放目录
├── music_library/            # 本地背景音乐曲库
├── requirements.txt          # Python依赖库列表
└── README.md                 # 项目说明文件
```

## 展望

- 增强对剪辑风格的控制，允许用户通过自然语言进行定制。
- 引入更深层次的语义理解模型（如场景识别、人物关系分析）。
- 优化处理性能，支持更高分辨率的视频。

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。
