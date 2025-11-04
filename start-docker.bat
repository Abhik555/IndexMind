@echo off
set IMAGE_NAME=index-mind
set CONTAINER_NAME=indexmind_container

echo 🔨 Building Docker image: %IMAGE_NAME% ...
docker build -t %IMAGE_NAME% .

echo Checking for NVIDIA GPU runtime...
docker info | findstr /R "Runtimes:.*nvidia" >nul
if %ERRORLEVEL%==0 (
    echo 🚀 Starting container with GPU support...
    docker run --rm -it ^
        --gpus all ^
        -p 8000:8000 ^
        --name %CONTAINER_NAME% ^
        %IMAGE_NAME%
) else (
    echo ⚠️  No NVIDIA runtime detected. Starting container without GPU...
    docker run --rm -it ^
        -p 8000:8000 ^
        --name %CONTAINER_NAME% ^
        %IMAGE_NAME%
)

pause
