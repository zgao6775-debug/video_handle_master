关键帧与摘要统一输出（含多语言/翻译）

功能
- 关键帧抽取：按场景切分，导出关键帧图片与时间点
- 统一报告：生成 report.json（关键帧 + 场景 + 转写 + 摘要 + 翻译信息）
- 多语言：Whisper 自动识别源语言；支持英文翻译（Whisper translate），以及可选的 DashScope 翻译
- 可选字幕：输出 SRT（源语言与英文）

使用
1) 生成报告与关键帧
python video-unified-output/unified_report.py --input d:/rubbish/video-handle/video-Motion-Analysis/example.mp4 --targets en --max-keyframes 12

2) 同时输出字幕
python video-unified-output/unified_report.py --input d:/rubbish/video-handle/video-Motion-Analysis/example.mp4 --targets en --with-srt

3) 启用 DashScope（更好的摘要/多语翻译）
python video-unified-output/unified_report.py --input d:/rubbish/video-handle/video-Motion-Analysis/example.mp4 --targets en,ja --use-dashscope

说明
- DashScope 需要环境变量 DASHSCOPE_API_KEY（或 Video-auto-editing/config.py 中的 API_KEY）且显式传入 --use-dashscope 才会调用
- 未开启 DashScope 时，也能生成关键帧与基础摘要；英文翻译使用 Whisper translate（只支持翻成英文）
