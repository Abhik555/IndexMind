# IndexMind Backend

### Steps to setup and run backend server

1. Using uv(Recommended)
    
    ```
    install uv
    clone repo
    uv init
    uv sync

    uvicorn backend:app
    ```
2. Using pip

    ```
    pip install -r requirements.txt
    uvicorn backend:app
    ```