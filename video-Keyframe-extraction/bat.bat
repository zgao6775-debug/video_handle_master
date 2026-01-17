@echo off
REM 切换到脚本所在目录
cd /d "D:\rubbish\video-handle\video-Keyframe-extraction\bin"

REM 第一步：运行 1.py
echo 正在运行 1.py ...
python 1.py
if %errorlevel% neq 0 (
    echo 运行 1.py 出错，已退出。
    pause
    exit /b %errorlevel%
)

REM 第二步：运行 2.py
echo 正在运行 2.py ...
python 2.py
if %errorlevel% neq 0 (
    echo 运行 2.py 出错，已退出。
    pause
    exit /b %errorlevel%
)

echo 全部执行完成 ✅
pause
