import os
import requests
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_from_drive(file_id: str, dest_path: Path) -> bool:
    """
    Download de arquivo do Google Drive com tratamento de erros
    """
    try:
        # Garante que a pasta existe
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        URL = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        
        logger.info(f"Iniciando download do arquivo {dest_path}")
        response = session.get(URL, params={'id': file_id}, stream=True)
        response.raise_for_status()  # Verifica se a resposta é válida
        
        token = next(
            (value for key, value in response.cookies.items() 
             if key.startswith('download_warning')),
            None
        )
        
        if token:
            response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
            response.raise_for_status()
        
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Download concluído: {dest_path}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Erro no download: {e}")
        return False
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        return False

def ensure_model(model_path: str, file_id: str) -> bool:
    """
    Garante que o modelo existe, baixando se necessário
    """
    path = Path(model_path)
    
    if path.exists():
        logger.info(f"Modelo já existe em: {path}")
        return True
        
    logger.info("Baixando modelo do Google Drive...")
    return download_from_drive(file_id, path)

# Configuração do modelo
MODEL_PATH = "ml_models/pha_classifier.pkl"
MODEL_ID = "1nrKe7WgEuih3OxmUrAeTwpoeJPEvNbef"

# Garante que o modelo está disponível
if not ensure_model(MODEL_PATH, MODEL_ID):
    logger.error("Falha ao garantir disponibilidade do modelo")
    exit(1)

# Configuração da API
import uvicorn
from fastapi import FastAPI
from app.api.endpoints import simulation

app = FastAPI(
    title="NASA Hackathon - Meteor Madness API",
    description="API para simulação de impacto de asteroides.",
    version="1.0.0"
)

app.include_router(simulation.router, prefix="/api/v1", tags=["Simulation"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Bem-vindo à API do Simulador de Impacto!"}

if __name__ == "__main__":
    logger.info("Iniciando servidor API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)