import numpy as np
import joblib
import pickle
import pandas as pd
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("ml_models/pha_classifier.pkl")

def load_model(model_path: Path):
    """Tenta carregar o modelo com diferentes métodos"""
    try:
        logger.info(f"Tentando carregar modelo de {model_path} com joblib...")
        return joblib.load(model_path)
    except Exception as e:
        logger.warning(f"Erro ao carregar com joblib: {e}")
        try:
            logger.info("Tentando carregar com pickle...")
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar com pickle: {e}")
            raise RuntimeError(f"Não foi possível carregar o modelo: {e}")

try:
    PHA_CLASSIFIER_MODEL = load_model(MODEL_PATH)
    logger.info("Modelo carregado com sucesso!")
except Exception as e:
    logger.error(f"Erro crítico ao carregar modelo: {e}")
    PHA_CLASSIFIER_MODEL = None

def predict_pha_risk(features_dict: dict) -> bool:
    """Usa o modelo carregado para prever se um asteroide é um PHA."""
    if PHA_CLASSIFIER_MODEL is None:
        logger.error("Modelo não está disponível para predição")
        return False
        
    try:
        features_order = [
            'absolute_magnitude_h', 'diameter_km_avg', 'velocity_km_s',
            'miss_distance_km', 'kinetic_energy_joules'
        ]
        
        input_df = pd.DataFrame([features_dict], columns=features_order)
        prediction = PHA_CLASSIFIER_MODEL.predict(input_df)
        
        return bool(prediction[0])
    except Exception as e:
        logger.error(f"Erro durante predição: {e}")
        return False

def calculate_damage_from_pair_model(
    diameter_km: float, 
    velocity_km_s: float, 
    impact_angle: float,
    densidade_rho: float,
    eficiencia_eta: float
    ) -> dict:
    """
    Calcula as consequências de um impacto usando as equações do modelo PAIR.
    """
    try:
        # 1. Calcular Energia de Impacto (E)
        # Primeiro, convertemos tudo para unidades do SI (metros, kg, segundos)
        raio_m = (diameter_km * 1000) / 2
        velocidade_m_s = velocity_km_s * 1000
        
        # Volume (assumindo uma esfera) e Massa
        volume_m3 = (4/3) * np.pi * (raio_m ** 3)
        massa_kg = densidade_rho * volume_m3
        
        # Energia Cinética em Joules, corrigida pelo ângulo de entrada
        energia_cinetica_j = 0.5 * massa_kg * (velocidade_m_s ** 2) * np.sin(np.radians(impact_angle))
        
        # Converter para Megatons de TNT (1 MT = 4.184e15 Joules)
        energia_E_megatons = energia_cinetica_j / 4.184e15

        # 2. Simplificar Altitude de Explosão (h)
        if diameter_km < 0.05: # Menos de 50 metros
            altitude_h_km = 5.0 # Assume explosão a 5km de altitude
        else:
            altitude_h_km = 0.0 # Assume impacto direto no solo

        # 3. Calcular Raio de Dano por Explosão
        E = energia_E_megatons
        h = altitude_h_km
        
        if h > 0: # Se for um airburst
            raio_blast_km = (2.09 * (h**-0.449) * (h**2) * (E**(-1/3))) + (5.08 * (E**(1/3)))
        else: # Se for impacto no solo
            raio_blast_km = 5.08 * (E**(1/3))

        # 4. Calcular Raio de Dano Térmico
        eta = eficiencia_eta
        Zi = 0.42 * (E**(1/6))
        r_sq = (eta * E) / (2 * np.pi * Zi)
        h_sq = h**2
        
        raio_thermal_km = np.sqrt(r_sq - h_sq) if r_sq > h_sq else 0.0
        
        # 5. Montar o dicionário de resultados
        results = {
            "energia_megatons": round(E, 2),
            "altitude_explosao_km": round(h, 2),
            "raio_dano_explosao_km": round(raio_blast_km, 2),
            "raio_dano_termico_km": round(raio_thermal_km, 2),
            "raio_dano_final_km": round(max(raio_blast_km, raio_thermal_km), 2),
            "kinetic_energy_joules": energia_cinetica_j
        }
        
        return results
    except Exception as e:
        logger.error(f"Erro no cálculo de dano: {e}")
        return None