"""
test_api.py
-----------
Script pour tester l'API FastAPI avec chargement automatique du pipeline.
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"

def wait_for_api(max_attempts=10):
    """Attend que l'API soit disponible"""
    print("⏳ Attente du démarrage de l'API...")
    for i in range(max_attempts):
        try:
            requests.get(f"{BASE_URL}/health/", timeout=2)
            print("✅ API disponible")
            return True
        except requests.exceptions.ConnectionError:
            if i < max_attempts - 1:
                time.sleep(1)
    return False


def load_pipeline():
    """Charge le pipeline via l'API"""
    print("\n" + "="*70)
    print("CHARGEMENT DU PIPELINE")
    print("="*70)
    
    response = requests.post(f"{BASE_URL}/load-pipeline/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        print("✅ Pipeline chargé avec succès")
        return True
    else:
        print("❌ Échec du chargement du pipeline")
        return False


def test_health():
    """Test du endpoint /health/"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Test du endpoint /predict/ avec un client"""
    print("\n" + "="*70)
    print("TEST 2: Prédiction pour un client unique (profil favorable)")
    print("="*70)
    
    # Client 1 : profil à faible risque
    client_data = {
        "Seniority": 15.0,
        "Home": 1.0,
        "Time": 60.0,
        "Age": 45.0,
        "Marital": 1.0,
        "Records": 0.0,
        "Job": 1.0,
        "Expenses": 50.0,
        "Income": 250.0,
        "Assets": 5000.0,
        "Debt": 0.0,
        "Amount": 1000.0,
        "Price": 1500.0
    }
    
    print("\n📋 Données du client:")
    print(f"  - Ancienneté: {client_data['Seniority']} ans")
    print(f"  - Âge: {client_data['Age']} ans")
    print(f"  - Revenu: {client_data['Income']}€")
    print(f"  - Actifs: {client_data['Assets']}€")
    print(f"  - Dette: {client_data['Debt']}€")
    print(f"  - Montant demandé: {client_data['Amount']}€")
    
    response = requests.post(f"{BASE_URL}/predict/", json=client_data)
    
    if response.status_code == 200:
        result = response.json()
        print("\n📊 Résultat de la prédiction:")
        print(f"  🎯 Décision: {result['decision']}")
        print(f"  📈 Confiance: {result['confidence']:.2%}")
        print(f"  ✅ P(Approuvé): {result['probability_approved']:.2%}")
        print(f"  ❌ P(Défaut): {result['probability_default']:.2%}")
    else:
        print(f"\n❌ Erreur (Status: {response.status_code}):")
        print(json.dumps(response.json(), indent=2))


def test_risky_client():
    """Test avec un client à haut risque"""
    print("\n" + "="*70)
    print("TEST 3: Prédiction pour un client à haut risque")
    print("="*70)
    
    # Client à haut risque
    risky_client = {
        "Seniority": 0.0,
        "Home": 0.0,
        "Time": 24.0,
        "Age": 22.0,
        "Marital": 0.0,
        "Records": 2.0,
        "Job": 0.0,
        "Expenses": 150.0,
        "Income": 100.0,
        "Assets": 0.0,
        "Debt": 500.0,
        "Amount": 5000.0,
        "Price": 6000.0
    }
    
    print("\n📋 Données du client:")
    print(f"  - Ancienneté: {risky_client['Seniority']} ans (nouveau)")
    print(f"  - Âge: {risky_client['Age']} ans (jeune)")
    print(f"  - Revenu: {risky_client['Income']}€ (faible)")
    print(f"  - Actifs: {risky_client['Assets']}€ (aucun)")
    print(f"  - Dette: {risky_client['Debt']}€ (élevée)")
    print(f"  - Montant demandé: {risky_client['Amount']}€ (élevé)")
    
    response = requests.post(f"{BASE_URL}/predict/", json=risky_client)
    
    if response.status_code == 200:
        result = response.json()
        print("\n📊 Résultat de la prédiction:")
        print(f"  🎯 Décision: {result['decision']}")
        print(f"  📈 Confiance: {result['confidence']:.2%}")
        print(f"  ✅ P(Approuvé): {result['probability_approved']:.2%}")
        print(f"  ❌ P(Défaut): {result['probability_default']:.2%}")
    else:
        print(f"\n❌ Erreur (Status: {response.status_code}):")
        print(json.dumps(response.json(), indent=2))


def test_batch_prediction():
    """Test du endpoint /predict-batch/ avec plusieurs clients"""
    print("\n" + "="*70)
    print("TEST 4: Prédiction pour plusieurs clients (batch)")
    print("="*70)
    
    batch_data = {
        "clients": [
            {
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
            },
            {
                "Seniority": 17.0,
                "Home": 1.0,
                "Time": 60.0,
                "Age": 58.0,
                "Marital": 1.0,
                "Records": 1.0,
                "Job": 0.0,
                "Expenses": 48.0,
                "Income": 131.0,
                "Assets": 0.0,
                "Debt": 0.0,
                "Amount": 1000.0,
                "Price": 1658.0
            },
            {
                "Seniority": 10.0,
                "Home": 0.0,
                "Time": 36.0,
                "Age": 46.0,
                "Marital": 0.0,
                "Records": 2.0,
                "Job": 1.0,
                "Expenses": 90.0,
                "Income": 200.0,
                "Assets": 3000.0,
                "Debt": 0.0,
                "Amount": 2000.0,
                "Price": 2985.0
            }
        ]
    }
    
    print(f"\n📋 Nombre de clients: {len(batch_data['clients'])}")
    
    response = requests.post(f"{BASE_URL}/predict-batch/", json=batch_data)
    
    if response.status_code == 200:
        result = response.json()
        print("\n📊 Résumé des prédictions:")
        print(f"  Total clients : {result['total_clients']}")
        print(f"  ✅ Approuvés  : {result['approved_count']}")
        print(f"  ❌ Refusés    : {result['rejected_count']}")
        
        print("\n📋 Détail des prédictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"\n  Client {i}:")
            print(f"    Décision    : {pred['decision']}")
            print(f"    Confiance   : {pred['confidence']:.2%}")
            print(f"    P(approuvé) : {pred['probability_approved']:.2%}")
    else:
        print(f"\n❌ Erreur (Status: {response.status_code}):")
        print(json.dumps(response.json(), indent=2))


def test_pipeline_info():
    """Test du endpoint /pipeline-info/"""
    print("\n" + "="*70)
    print("TEST 5: Informations sur le pipeline")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/pipeline-info/")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Pipeline chargé")
        print(f"  Chargé à: {result.get('loaded_at', 'N/A')}")
        print(f"  Nombre d'étapes: {result.get('number_of_steps', 'N/A')}")
        print(f"  Étapes: {', '.join(result.get('pipeline_steps', []))}")
        print(f"  Classifieur: {result.get('classifier_type', 'N/A')}")
    else:
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


def run_all_tests():
    """Exécute tous les tests"""
    print("\n" + "="*70)
    print(" 🧪 TESTS DE L'API CREDIT SCORING")
    print("="*70)
    
    try:
        # Attendre que l'API soit disponible
        if not wait_for_api():
            print("\n❌ ERREUR: Impossible de se connecter à l'API")
            print("   Assurez-vous que l'API est démarrée avec : uvicorn api:app --reload")
            return
        
        # Vérifier l'état initial
        test_health()
        
        # Charger le pipeline
        if not load_pipeline():
            print("\n⚠️ Le pipeline n'a pas pu être chargé.")
            print("   Vérifiez que le fichier 'credit_scoring_pipeline.pkl' existe.")
            print("   Exécutez d'abord 'python main.py' pour générer le pipeline.")
            return
        
        # Vérifier que le pipeline est bien chargé
        test_health()
        
        # Exécuter les tests de prédiction
        test_single_prediction()
        test_risky_client()
        test_batch_prediction()
        test_pipeline_info()
        
        print("\n" + "="*70)
        print(" ✅ TOUS LES TESTS TERMINÉS AVEC SUCCÈS")
        print("="*70)
    
    except requests.exceptions.ConnectionError:
        print("\n❌ ERREUR: Impossible de se connecter à l'API")
        print("   Assurez-vous que l'API est démarrée avec : uvicorn api:app --reload")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")


if __name__ == "__main__":
    run_all_tests()
