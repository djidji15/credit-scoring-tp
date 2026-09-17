"""
api.py
------
API FastAPI pour le scoring de crédit.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pickle
import os
from datetime import datetime

# ============================================================
# INITIALISATION DE L'API
# ============================================================

app = FastAPI(
    title="Credit Scoring API",
    description="API de prédiction de risque de crédit basée sur un pipeline ML",
    version="1.0.0"
)

# Variable globale pour stocker le pipeline
pipeline = None
pipeline_loaded_at = None

# ============================================================
# MODÈLES DE DONNÉES (PYDANTIC)
# ============================================================

class ClientFeatures(BaseModel):
    """
    Modèle représentant les caractéristiques d'un client.
    Les 13 features du dataset credit_scoring.
    """
    Seniority: float = Field(..., description="Ancienneté (années)", example=9.0)
    Home: float = Field(..., description="Propriétaire (1) ou non (0)", example=1.0)
    Time: float = Field(..., description="Durée du prêt (mois)", example=60.0)
    Age: float = Field(..., description="Âge du client", example=30.0)
    Marital: float = Field(..., description="Statut marital", example=0.0)
    Records: float = Field(..., description="Antécédents de crédit", example=1.0)
    Job: float = Field(..., description="Type d'emploi", example=1.0)
    Expenses: float = Field(..., description="Dépenses mensuelles", example=73.0)
    Income: float = Field(..., description="Revenu mensuel", example=129.0)
    Assets: float = Field(..., description="Actifs", example=0.0)
    Debt: float = Field(..., description="Dette", example=0.0)
    Amount: float = Field(..., description="Montant du prêt", example=800.0)
    Price: float = Field(..., description="Prix du bien", example=846.0)

    class Config:
        schema_extra = {
            "example": {
                "Seniority": 9.0,
                "Home": 1.0,
                "Time": 60.0,
                "Age": 30.0,
                "Marital": 0.0,
                "Records": 1.0,
                "Job": 1.0,
                "Expenses": 73.0,
                "Income": 129.0,
                "Assets": 0.0,
                "Debt": 0.0,
                "Amount": 800.0,
                "Price": 846.0
            }
        }


class BatchClientFeatures(BaseModel):
    """
    Modèle pour prédire plusieurs clients en une seule requête.
    """
    clients: List[ClientFeatures]


class PredictionResponse(BaseModel):
    """
    Modèle de réponse pour une prédiction unique.
    """
    prediction: int = Field(..., description="0 = Refusé, 1 = Approuvé")
    prediction_label: str = Field(..., description="Label textuel de la prédiction")
    probability_default: float = Field(..., description="Probabilité de défaut (classe 0)")
    probability_approved: float = Field(..., description="Probabilité d'approbation (classe 1)")
    confidence: float = Field(..., description="Niveau de confiance (max des probabilités)")
    decision: str = Field(..., description="Décision finale")


class BatchPredictionResponse(BaseModel):
    """
    Modèle de réponse pour plusieurs prédictions.
    """
    predictions: List[PredictionResponse]
    total_clients: int
    approved_count: int
    rejected_count: int


class HealthResponse(BaseModel):
    """
    Modèle de réponse pour le health check.
    """
    status: str
    pipeline_loaded: bool
    pipeline_file: str
    loaded_at: Optional[str]


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def load_pipeline_from_file(filename: str = 'credit_scoring_pipeline.pkl'):
    """
    Charge le pipeline depuis un fichier pickle.
    """
    global pipeline, pipeline_loaded_at
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Le fichier {filename} n'existe pas")
    
    with open(filename, 'rb') as f:
        pipeline = pickle.load(f)
    
    pipeline_loaded_at = datetime.now().isoformat()
    print(f"✅ Pipeline chargé depuis {filename}")


def features_to_array(client: ClientFeatures) -> np.ndarray:
    """
    Convertit un objet ClientFeatures en array numpy dans le bon ordre.
    """
    return np.array([[
        client.Seniority,
        client.Home,
        client.Time,
        client.Age,
        client.Marital,
        client.Records,
        client.Job,
        client.Expenses,
        client.Income,
        client.Assets,
        client.Debt,
        client.Amount,
        client.Price
    ]])


def make_prediction(X: np.ndarray) -> PredictionResponse:
    """
    Effectue une prédiction avec le pipeline et retourne une réponse structurée.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline non chargé")
    
    # Prédiction
    prediction = int(pipeline.predict(X)[0])
    
    # Probabilités
    if hasattr(pipeline, 'predict_proba'):
        probas = pipeline.predict_proba(X)[0]
        prob_default = float(probas[0])
        prob_approved = float(probas[1])
    else:
        # Fallback si le modèle ne supporte pas predict_proba
        prob_default = 1.0 if prediction == 0 else 0.0
        prob_approved = 1.0 if prediction == 1 else 0.0
    
    confidence = max(prob_default, prob_approved)
    
    # Labels
    prediction_label = "Approuvé" if prediction == 1 else "Refusé"
    
    if prediction == 1:
        if confidence >= 0.8:
            decision = "✅ CRÉDIT APPROUVÉ - Risque faible"
        elif confidence >= 0.6:
            decision = "✅ CRÉDIT APPROUVÉ - Risque modéré"
        else:
            decision = "⚠️ CRÉDIT APPROUVÉ - Risque élevé (surveillance recommandée)"
    else:
        if confidence >= 0.8:
            decision = "❌ CRÉDIT REFUSÉ - Risque très élevé"
        elif confidence >= 0.6:
            decision = "❌ CRÉDIT REFUSÉ - Risque élevé"
        else:
            decision = "❌ CRÉDIT REFUSÉ - Décision limite (analyse manuelle recommandée)"
    
    return PredictionResponse(
        prediction=prediction,
        prediction_label=prediction_label,
        probability_default=prob_default,
        probability_approved=prob_approved,
        confidence=confidence,
        decision=decision
    )


# ============================================================
# ÉVÉNEMENTS DE DÉMARRAGE
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    Charge automatiquement le pipeline au démarrage de l'API.
    """
    try:
        load_pipeline_from_file('credit_scoring_pipeline.pkl')
        print("🚀 API démarrée avec succès")
    except Exception as e:
        print(f"⚠️ Avertissement : impossible de charger le pipeline au démarrage: {e}")
        print("   Vous pouvez charger manuellement via /load-pipeline/")


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint racine - informations sur l'API.
    """
    return {
        "message": "Credit Scoring API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health/",
            "predict_single": "/predict/",
            "predict_batch": "/predict-batch/",
            "load_pipeline": "/load-pipeline/"
        }
    }


@app.get("/health/", response_model=HealthResponse, tags=["Status"])
async def health_check():
    """
    Vérifie l'état de l'API et du pipeline.
    """
    return HealthResponse(
        status="healthy" if pipeline is not None else "pipeline not loaded",
        pipeline_loaded=pipeline is not None,
        pipeline_file="credit_scoring_pipeline.pkl",
        loaded_at=pipeline_loaded_at
    )


@app.post("/load-pipeline/", tags=["Admin"])
async def load_pipeline_endpoint(filename: str = "credit_scoring_pipeline.pkl"):
    """
    Charge ou recharge le pipeline depuis un fichier pickle.
    """
    try:
        load_pipeline_from_file(filename)
        return {
            "status": "success",
            "message": f"Pipeline chargé depuis {filename}",
            "loaded_at": pipeline_loaded_at
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement: {str(e)}")


@app.post("/predict/", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(client: ClientFeatures):
    """
    Prédit le risque de crédit pour un seul client.
    
    Retourne :
    - prediction : 0 (refusé) ou 1 (approuvé)
    - probabilités pour chaque classe
    - décision avec niveau de confiance
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503, 
            detail="Pipeline non chargé. Appelez d'abord /load-pipeline/"
        )
    
    try:
        X = features_to_array(client)
        response = make_prediction(X)
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict-batch/", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchClientFeatures):
    """
    Prédit le risque de crédit pour plusieurs clients en une seule requête.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline non chargé. Appelez d'abord /load-pipeline/"
        )
    
    try:
        predictions = []
        
        for client in batch.clients:
            X = features_to_array(client)
            pred = make_prediction(X)
            predictions.append(pred)
        
        approved_count = sum(1 for p in predictions if p.prediction == 1)
        rejected_count = len(predictions) - approved_count
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_clients=len(predictions),
            approved_count=approved_count,
            rejected_count=rejected_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.get("/pipeline-info/", tags=["Admin"])
async def pipeline_info():
    """
    Retourne des informations sur le pipeline chargé.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline non chargé")
    
    steps = [step[0] for step in pipeline.steps]
    
    info = {
        "pipeline_loaded": True,
        "loaded_at": pipeline_loaded_at,
        "pipeline_steps": steps,
        "number_of_steps": len(steps)
    }
    
    # Informations sur le classifieur
    if 'classifier' in pipeline.named_steps:
        classifier = pipeline.named_steps['classifier']
        info["classifier_type"] = type(classifier).__name__
        
        if hasattr(classifier, 'get_params'):
            info["classifier_params"] = classifier.get_params()
    
    return info


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
