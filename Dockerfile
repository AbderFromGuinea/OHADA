# ---------- STAGE 1 : Build & install dependencies ----------
FROM python:3.10-slim AS builder

WORKDIR /app

# Copie uniquement les requirements pour installer les libs
COPY requirements.txt .

# Installer gcc et les outils nécessaires temporairement
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --target=/install -r requirements.txt \
 && apt-get purge -y --auto-remove gcc build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /root/.cache

# ---------- STAGE 2 : Image finale propre et légère ----------
FROM python:3.10-slim

WORKDIR /app

# Copier les dépendances installées depuis le premier stage
COPY --from=builder /install /usr/local/lib/python3.10/site-packages

# Copier ton projet
COPY . .

# Exposer le port (utile sur Railway ou autre)
EXPOSE 8000

# Commande de démarrage de l'API
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
