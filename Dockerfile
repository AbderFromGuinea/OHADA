# Stage 1: Build environment
FROM python:3.11-slim as builder

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Crée un venv temporaire pour builder les paquets
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime minimal
FROM python:3.11-slim

WORKDIR /app

# Copie l'environnement virtuel depuis le builder
COPY --from=builder /opt/venv /opt/venv

# Copie le code
COPY . /app

# Active le venv par défaut
ENV PATH="/opt/venv/bin:$PATH"

# Port Railway
ENV PORT=8000

# Commande de démarrage
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
