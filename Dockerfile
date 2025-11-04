FROM python:3.10.19-trixie

WORKDIR /IndexMind

RUN pip install --no-cache-dir uv

COPY . .

RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN uv pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn" , "backend:app"]

