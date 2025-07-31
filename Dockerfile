# --------- STAGE 1: Build Dependencies ---------
FROM python:3.10-slim AS builder

WORKDIR /app

# Préinstaller pip et venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && python -m venv /opt/venv \
    && . /opt/venv/bin/activate

# Copier uniquement requirements
COPY requirements.txt .

# Installer les dépendances
RUN . /opt/venv/bin/activate && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 🧹 Nettoyage : réduire la taille de l'image
RUN apt-get purge -y --auto-remove gcc build-essential

# --------- STAGE 2: Runtime ---------
FROM python:3.10-slim

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Installer uniquement ce qui est nécessaire pour exécuter
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .

# Exposer le port Railway
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
