FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy the project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY conf/ ./conf/
COPY main.py ./

# Sync dependencies globally
RUN uv sync --system

ENTRYPOINT ["python", "main.py"]
