# IndexMind Backend

## 🐳 Run with Docker (Recommended)

### 1️⃣ Build the image

```bash
docker build -t index-mind .
```

### 2️⃣ Run the container

```bash
docker run -p 8000:8000 index-mind
```

You app should be live on http://localhost:8000

---

## 💻 Run Locally (Without Docker)

If you prefer to run it on your system directly (using **requirements.txt**):

### 1️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
# or
.venv\Scripts\activate     # (Windows)
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
uvicorn backend:app
```