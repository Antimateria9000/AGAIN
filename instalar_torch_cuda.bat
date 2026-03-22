@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo No existe .venv\Scripts\python.exe
  pause
  exit /b 1
)

echo Instalando torch con CUDA en el entorno virtual...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip uninstall -y torch
".venv\Scripts\python.exe" -m pip install -r requirements-torch-cu128.txt

echo Verificando runtime CUDA...
".venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"

pause
