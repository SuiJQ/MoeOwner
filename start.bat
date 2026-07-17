@echo off
chcp 65001 >nul
title MoeOwner 推理引擎
setlocal enabledelayedexpansion

REM ──────────────────────────────────────────────────────────────────────────
REM MoeOwner — MoE 异构推理引擎 Windows 一键启动
REM
REM   使用方法：
REM     1. 下载你喜欢的 GGUF 格式模型文件
REM     2. 放入本目录下的 models\ 文件夹
REM     3. 改名为 Model.gguf
REM     4. 双击本文件即可运行
REM
REM   例如： models\Model.gguf
REM ──────────────────────────────────────────────────────────────────────────

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM ──────────────────────────────────────────────────────────────────────────
REM 颜色输出 (Windows 10+ 支持 ANSI)
REM ──────────────────────────────────────────────────────────────────────────
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "CYAN=[96m"
set "NC=[0m"

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║               MoeOwner  MoE 推理引擎                      ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM ══════════════════════════════════════════════════════════════════
REM  STEP 1 — 检查模型文件
REM ══════════════════════════════════════════════════════════════════

set MODEL_FILE=models\Model.gguf

if not exist "%MODEL_FILE%" (
    echo %RED%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%NC%
    echo %RED%  ✗ 未找到模型文件!                                    %NC%
    echo %RED%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%NC%
    echo.
    echo 请按以下步骤操作：
    echo.
    echo  1. 下载一个 GGUF 格式的模型（例如从 HuggingFace）
    echo     推荐: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
    echo     https://huggingface.co/Qwen
    echo.
    echo  2. 在本目录下创建 models\ 文件夹（如果还没有）
    echo.
    echo  3. 把下载的 .gguf 文件放入 models\ 并改名为:
    echo        %CYAN%Model.gguf%NC%
    echo.
    echo  4. 重新双击本文件启动
    echo.
    echo  ── 当前目录结构应该像这样 ──
    echo    MoeOwner\
    echo    ├── main.py
    echo    ├── start.bat      ＜── 就是这个文件
    echo    ├── models\
    echo    │   └── Model.gguf  ＜── 你的模型
    echo    ├── scheduler.py
    echo    └── ...
    echo.
    pause
    exit /b 1
)

echo %GREEN%[✓]%NC% 模型文件: %MODEL_FILE%
for %%F in ("%MODEL_FILE%") do echo       ├── 文件名: %%~nxF
for %%F in ("%MODEL_FILE%") do echo       └── 大小: %%~zF 字节

echo.

REM ══════════════════════════════════════════════════════════════════
REM  STEP 2 — Python 检测
REM ══════════════════════════════════════════════════════════════════

set PYTHON=
where python 2>nul >nul
if %errorlevel% equ 0 (
    set PYTHON=python
    goto :python_ok
)

set "PYPATHS=C:\Python313\python.exe;C:\Python312\python.exe;C:\Program Files\Python313\python.exe;C:\Program Files\Python312\python.exe"
for %%p in (%PYPATHS%) do (
    if exist "%%p" (
        set PYTHON=%%p
        goto :python_ok
    )
)

echo %RED%[✗] 未找到 Python!%NC%
echo     请先安装 Python 3.12+
echo     下载地址: https://www.python.org/downloads/
echo     安装时请勾选 "Add Python to PATH"
pause
exit /b 1

:python_ok
for /f "tokens=*" %%i in ('%PYTHON% --version 2^>nul') do echo %GREEN%[✓]%NC% %%i

REM ══════════════════════════════════════════════════════════════════
REM  STEP 3 — 虚拟环境 + 依赖
REM ══════════════════════════════════════════════════════════════════

if not exist ".venv" (
    echo %YELLOW%[~]%NC% 正在创建虚拟环境...
    %PYTHON% -m venv .venv >nul 2>&1
    echo %GREEN%[✓]%NC% 虚拟环境已创建
)

call .venv\Scripts\activate.bat

REM PyTorch
python -c "import torch" 2>nul
if %errorlevel% neq 0 (
    echo %YELLOW%[~]%NC% 正在安装 PyTorch（约需几分钟）...
    pip install --quiet --upgrade pip
    pip install --quiet torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
    if !errorlevel! neq 0 (
        echo %YELLOW%[~]%NC% CUDA 版安装失败,尝试 CPU 版...
        pip install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    )
    echo %GREEN%[✓]%NC% PyTorch 安装完成
) else (
    echo %GREEN%[✓]%NC% PyTorch 已就绪
)

REM ══════════════════════════════════════════════════════════════════
REM  STEP 4 — CUDA 检测
REM ══════════════════════════════════════════════════════════════════

python -c "import torch; g=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; print(g)" 2>nul >_gpu.tmp
set /p GPU_NAME=<_gpu.tmp
del _gpu.tmp 2>nul

if "%GPU_NAME%"=="N/A" (
    echo %YELLOW%[!]%NC% 未检测到 CUDA GPU，将使用 CPU 推理（速度较慢）
) else (
    echo %GREEN%[✓]%NC% GPU: %GPU_NAME%
)

echo.

REM ══════════════════════════════════════════════════════════════════
REM  STEP 5 — 启动引擎
REM ══════════════════════════════════════════════════════════════════

echo ═══════════════════════════════════════════════════════════════
echo  启动中...
echo  模型: %MODEL_FILE%
echo.
echo  按 Ctrl+C 可安全停止引擎
echo ═══════════════════════════════════════════════════════════════
echo.

python main.py --gguf "%MODEL_FILE%"

set EXIT_CODE=%errorlevel%
if %EXIT_CODE% neq 0 (
    echo.
    echo %RED%[✗] 引擎异常退出，错误码: %EXIT_CODE%%NC%
    echo     请检查上面输出的错误信息
)
pause
exit /b %EXIT_CODE%
