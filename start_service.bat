@echo off
setlocal
chcp 65001>nul
set PYTHONUTF8=1

set EMBEDDING_MODEL_DIR=%EMBEDDING_MODEL_DIR%
if "%EMBEDDING_MODEL_DIR%"=="" set EMBEDDING_MODEL_DIR=%~dp0Models\embeddinggemma-2-onnx

if not "%PYTHON_EXE%"=="" if exist "%PYTHON_EXE%" set "PYTHON_CMD=%PYTHON_EXE%"
if not defined PYTHON_CMD if not "%CONDA_PREFIX%"=="" if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_CMD=%CONDA_PREFIX%\python.exe"
if not defined PYTHON_CMD if not "%VIRTUAL_ENV%"=="" if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PYTHON_CMD=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined PYTHON_CMD for /f "delims=" %%I in ('where python 2^>nul') do (
	echo %%~fI | findstr /I /C:"WindowsApps\python.exe" >nul
	if errorlevel 1 if not defined PYTHON_CMD set "PYTHON_CMD=%%~fI"
)
if not defined PYTHON_CMD for /f "delims=" %%I in ('where python 2^>nul') do if not defined PYTHON_CMD set "PYTHON_CMD=%%~fI"
if not defined PYTHON_CMD (
	echo 未找到可用的 Python 解释器，请设置 PYTHON_EXE 后重试。
	pause
	exit /b 1
)

echo 启动 EmbeddingGemma2 ONNX REST Service...
echo.
echo 服务将在以下地址启动:
echo   - 主页: http://localhost:8000/
echo   - Swagger UI (离线): http://localhost:8000/docs
echo   - ReDoc: http://localhost:8000/redoc
echo   - 健康检查: http://localhost:8000/health
echo.
echo 当前模型目录:
echo   - Embedding: %EMBEDDING_MODEL_DIR%
echo 当前 Python:
echo   - Executable: %PYTHON_CMD%
echo.
echo %PYTHON_CMD% | findstr /I /C:"WindowsApps\python.exe" >nul
if %ERRORLEVEL% EQU 0 (
    echo 警告: 当前使用的是 Windows Store Python，可能无法加载期望的 CUDA 环境。
    echo 建议先设置 PYTHON_EXE，或在正确的 conda/venv 环境中运行此脚本。
    echo.
)
echo.
echo ONNX 环境检测结果:
"%PYTHON_CMD%" -c "import sys, onnxruntime as ort; from onnx_provider_utils import choose_execution_providers, provider_chain_to_string; providers, reason, available = choose_execution_providers(); print('  - Python:', sys.executable); print('  - ONNX Runtime:', ort.__version__); print('  - Available Providers:', available); print('  - Selected Providers:', provider_chain_to_string(providers)); print('  - Reason:', reason)"
if %ERRORLEVEL% NEQ 0 (
	echo ONNX 环境检测失败，已停止启动。
	pause
	exit /b 1
)
echo.
echo 按 Ctrl+C 停止服务
echo.

cd /d "%~dp0"
"%PYTHON_CMD%" main.py
pause
