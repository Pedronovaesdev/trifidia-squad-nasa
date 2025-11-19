# ☄️ **AstroView API** - Backend & Simulation Engine

<div align="center">

![AstroView Backend Banner](https://img.shields.io/badge/AstroView-Backend%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA%20Space%20Apps-Challenge%202025-blue?style=flat-square&logo=nasa)](https://www.spaceappschallenge.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini%202.0-8E75B2?style=flat-square&logo=google)](https://ai.google.dev/)

**O núcleo computacional por trás do sistema de defesa planetária AstroView.**

*Uma API de alta performance que integra modelos físicos de impacto, análise geoespacial de risco e inteligência artificial generativa.*

[🚀 **Live API**](https://api.astroview.space/docs) · [📖 **Documentação**](https://github.com/julianaivo/trifidia-squad-nasa-backend/wiki) · [🐛 **Reportar Bug**](https://github.com/julianaivo/trifidia-squad-nasa-backend/issues)

</div>

---

## 🌟 **Visão Geral**

A **AstroView API** é o "cérebro" da plataforma. Desenvolvida em **FastAPI**, ela é responsável por orquestrar simulações complexas de impacto de asteroides em tempo real. O sistema combina equações físicas validadas (Modelo PAIR) com aprendizado de máquina para prever riscos e utiliza LLMs (Google Gemini) para traduzir dados técnicos em relatórios de emergência compreensíveis.

### 🎯 **Principais Responsabilidades**
1.  **Motor de Física**: Calcula energia cinética (Megatons), crateras e ondas de choque baseando-se em densidade, velocidade e ângulo.
2.  **Inteligência Artificial**: Integração com **Google Gemini 2.0 Flash** para gerar briefings de emergência e atuar como assistente virtual.
3.  **Análise Geoespacial**: Processamento de dados raster (`.tif`) para estimar população afetada e impacto econômico no local da colisão.
4.  **ML Pipeline**: Download automático e execução de modelos `.pkl` para classificação de Asteroides Potencialmente Perigosos (PHAs).

---

## ✨ **Funcionalidades Chave**

### 💥 **Simulação de Impacto (Physics Engine)**
- **Módulo**: `app/services/simulation_engine.py`
- Implementa o cálculo de **Airburst** vs **Impacto no Solo**.
- Calcula raios de dano térmico e sobrepressão (blast).
- **Segurança**: Regra híbrida que combina predição de ML com verificação física (limiar de 500 MT).

### 🤖 **Assistente de Defesa Planetária (IA)**
- **Módulo**: `app/services/llm_services.py`
- Gera relatórios de risco estruturados (Resumo, Análise Humana, Recomendações).
- Endpoint de Chatbot (`/ask-agent`) contextualizado com os dados da simulação atual.

### 🌍 **Inteligência Geográfica**
- **Módulo**: `app/services/geo_analysis.py`
- Utiliza `Rasterio` e `GeoPandas` para cruzar o raio de impacto com mapas de densidade populacional.
- Estima prejuízos financeiros baseados na população atingida.

---

## 🛠️ **Tech Stack**

<div align="center">

| **Framework** | **Data Science** | **Geo Spatial** | **AI & ML** |
|:---:|:---:|:---:|:---:|
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | ![Rasterio](https://img.shields.io/badge/Rasterio-Geo-green?style=flat-square) | ![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=flat-square&logo=google&logoColor=white) |
| **Uvicorn** | **NumPy** | **Shapely** | **Scikit-Learn** |

</div>

---

## 🚀 **Instalação e Configuração**

### **Pré-requisitos**
- Python 3.10 ou superior
- Chave de API do Google (Gemini)

### **Passo a Passo**

1. **Clone o repositório**
   ```bash
   git clone [https://github.com/julianaivo/trifidia-squad-nasa-backend.git](https://github.com/julianaivo/trifidia-squad-nasa-backend.git)
   cd trifidia-squad-nasa-backend


2.  **Crie o Ambiente Virtual**

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Instale as Dependências**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure as Variáveis de Ambiente**
    Crie um arquivo `.env` na raiz do projeto:

    ```env
    GOOGLE_API_KEY="sua_chave_api_google_aqui"
    ```

5.  **Dados de População**
    O sistema espera um arquivo de dados populacionais. Certifique-se de que o arquivo existe em:
    `data/population/data.tif`

6.  **Execute o Servidor**
    O sistema baixará automaticamente o modelo de ML do Google Drive na primeira execução.

    ```bash
    python main.py
    # Ou via uvicorn diretamente
    # uvicorn main:app --reload
    ```

-----

## 🔌 **Endpoints da API**

A documentação interativa (Swagger UI) está disponível em `http://localhost:8000/docs`.

### **1. Executar Simulação**

`POST /api/v1/simulate`

Calcula os danos físicos e gera o relatório de IA.

**Input Exemplo:**

```json
{
  "diameter_km": 0.4,
  "velocity_km_s": 25,
  "impact_angle": 45,
  "impact_lat": -10.9,
  "impact_lng": -37.0,
  "asteroid_type": "Crystalline Rock"
}
```

### **2. Chat com Agente**

`POST /api/v1/ask-agent`

Permite conversar com a IA sobre os resultados da simulação.

**Input Exemplo:**

```json
{
  "question": "Quais seriam os efeitos de longo prazo deste impacto?",
  "context": { ... dados da simulação ... }
}
```

-----

## 📂 **Estrutura do Projeto**

```bash
trifidia-squad-nasa-backend/
├── app/
│   ├── api/
│   │   └── endpoints/    # Rotas (simulation.py)
│   ├── core/
│   │   ├── config.py     # Configurações e .env
│   │   └── constants.py  # Constantes físicas e cenários
│   ├── models/           # Schemas Pydantic (Input/Output)
│   └── services/
│       ├── simulation_engine.py # Física e ML
│       ├── llm_services.py      # Integração Google Gemini
│       └── geo_analysis.py      # Análise de População/Raster
├── ml_models/            # Modelos .pkl (baixados automaticamente)
├── data/                 # Dados geoespaciais (.tif)
├── main.py               # Entrypoint da aplicação
├── requirements.txt      # Dependências do Python
└── .env                  # Variáveis sensíveis
```

-----

## 🤝 **Contribuindo**

Contribuições são bem-vindas\! Se você tem melhorias para o modelo físico ou novos prompts para a IA:

1.  Faça um Fork do projeto
2.  Crie sua Feature Branch (`git checkout -b feature/MelhoriaFisica`)
3.  Commit suas mudanças (`git commit -m 'Add: Cálculo de Tsunami'`)
4.  Push para a Branch (`git push origin feature/MelhoriaFisica`)
5.  Abra um Pull Request

-----

**© 2025 Team Trifidia Squad | NASA Space Apps Challenge**


