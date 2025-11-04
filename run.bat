@echo off
set IMAGE_NAME=index-mind
set CONTAINER_NAME=indexmind_container

echo 🚀 Starting container from image: %IMAGE_NAME%

echo Checking for NVIDIA GPU runtime...
docker info | findstr /R "Runtimes:.*nvidia" >nul
if %ERRORLEVEL%==0 (
    echo ✅ NVIDIA runtime detected — running with GPU support...
    docker run --rm -it ^
        --gpus all ^
        -p 8000:8000 ^
        --name %CONTAINER_NAME% ^
        %IMAGE_NAME%
) else (
    echo ⚠️  NVIDIA runtime not found — running without GPU support...
    docker run --rm -it ^
        -p 8000:8000 ^
        --name %CONTAINER_NAME% ^
        %IMAGE_NAME%
)

pause
