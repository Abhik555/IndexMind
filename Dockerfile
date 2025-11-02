FROM python:3.10.19-trixie

WORKDIR /IndexMind

RUN pip install --no-cache-dir uv

COPY . .

RUN if [ -f pyproject.toml ]; then \
        uv sync --locked; \
    elif [ -f requirements.txt ]; then \
        uv pip install -r requirements.txt; \
    else \
        echo "No dependency file found"; \
    fi

EXPOSE 8000

CMD ["uvicorn" , "main:app"]

