"""
main.py
-------
Script principal pour exécuter l'orchestration complète du pipeline.
"""

import pandas as pd
import numpy as np
from utils import pipeline_generation_train_test_split, load_pipeline

# Chargement des données
df = pd.read_csv('../credit_scoring.csv', sep=';')

print("="*70)
print(" LANCEMENT DE L'ORCHESTRATION COMPLÈTE")
print("="*70)

# ===== ORCHESTRATION COMPLÈTE =====
pipeline, results = pipeline_generation_train_test_split(
    df=df,
    target_column='Status',
    test_size=0.5,
    random_state=1,
    use_scaler=True,
    use_pca=True,
    n_components=3,
    auto_select_features=True,
    n_features_to_select=None,  # Auto
    classifier_type='MLP',
    tune_hyperparameters=True,
    param_grid=None,  # Grille par défaut
    output_filename='credit_scoring_pipeline.pkl',
    verbose=True
)

# ===== AFFICHAGE DES FEATURES SÉLECTIONNÉES =====
if 'feature_selection' in pipeline.named_steps:
    feature_selector = pipeline.named_steps['feature_selection']
    selected_indices = feature_selector.selected_indices_
    
    original_features = df.drop(columns=['Status']).columns.tolist()
    feature_names = original_features + ["PCA1", "PCA2", "PCA3"]
    
    print("\n" + "="*70)
    print(f" FEATURES SÉLECTIONNÉES ({len(selected_indices)}/{len(feature_names)})")
    print("="*70)
    for rank, idx in enumerate(selected_indices, 1):
        print(f"  {rank}. {feature_names[idx]}")

# ===== TEST DE CHARGEMENT =====
print("\n" + "="*70)
print(" TEST DE CHARGEMENT DU PIPELINE")
print("="*70)

loaded_pipeline = load_pipeline('credit_scoring_pipeline.pkl')

# Test sur quelques exemples
X_test_sample = df.drop(columns=['Status']).values[:5]
predictions = loaded_pipeline.predict(X_test_sample)
probas = loaded_pipeline.predict_proba(X_test_sample)

print("\n📊 Prédictions sur 5 clients :")
print("-" * 70)
for i in range(len(predictions)):
    statut = "✅ APPROUVÉ" if predictions[i] == 1 else "❌ REFUSÉ"
    print(f"Client {i+1}: {statut} | P(défaut)={probas[i][0]:.2%} | P(ok)={probas[i][1]:.2%}")

print("\n" + "="*70)
print(" 🎉 PROCESSUS TERMINÉ")
print("="*70)
