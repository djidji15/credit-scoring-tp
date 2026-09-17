"""
main_cv.py
----------
Script principal pour exécuter la comparaison complète avec validation croisée.
"""

import pandas as pd
from cross_validation import (
    run_classifiers_cv,
    create_classifiers_dict,
    pipeline_generation_cv
)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

df = pd.read_csv('../credit_scoring.csv', sep=';')

print("="*80)
print(" COMPARAISON EXHAUSTIVE DES ALGORITHMES - VALIDATION CROISÉE")
print("="*80)

# ============================================================
# OPTION 1 : Comparaison simple (sans créer de pipeline)
# ============================================================

print("\n" + "="*80)
print(" OPTION 1 : COMPARAISON RAPIDE DES ALGORITHMES")
print("="*80)

X = df.drop(columns=['Status']).to_numpy()
y = df['Status'].values

# Créer le dictionnaire de classifieurs
classifiers = create_classifiers_dict(random_state=1)

# Exécuter la comparaison
results = run_classifiers_cv(X, y, classifiers, cv=10, verbose=True)

# Sauvegarder les résultats
results.to_csv('comparison_results.csv', index=False)
print("\n💾 Résultats sauvegardés dans 'comparison_results.csv'")

# ============================================================
# OPTION 2 : Orchestration complète avec pipeline final
# ============================================================

print("\n\n" + "="*80)
print(" OPTION 2 : ORCHESTRATION COMPLÈTE AVEC SÉLECTION DU MEILLEUR ALGORITHME")
print("="*80)

final_pipeline, results_complete = pipeline_generation_cv(
    df=df,
    target_column='Status',
    test_size=0.5,
    random_state=1,
    cv_folds=10,
    compare_preprocessing=True,
    use_scaler=True,
    use_pca=True,
    n_components=3,
    auto_select_features=True,
    n_features_to_select=None,
    tune_hyperparameters=True,
    output_filename='best_pipeline_cv.pkl',
    verbose=True
)

# ============================================================
# AFFICHAGE DES RÉSULTATS FINAUX
# ============================================================

print("\n" + "="*80)
print(" 📊 RÉSUMÉ COMPLET DES RÉSULTATS")
print("="*80)

print(f"\n🏆 Meilleur algorithme : {results_complete['best_algorithm']['name']}")
print(f"   Type : {results_complete['best_algorithm']['type']}")
print(f"   Score CV : {results_complete['best_algorithm']['custom_score_mean']:.3f} ± {results_complete['best_algorithm']['custom_score_std']:.3f}")

if 'feature_selection' in results_complete:
    print(f"\n📌 Features sélectionnées ({results_complete['feature_selection']['best_k']}) :")
    for feat in results_complete['feature_selection']['selected_features']:
        print(f"   - {feat}")

print("\n📈 Performances finales sur le test set :")
print(f"   Accuracy  : {results_complete['final_performance']['accuracy']:.3f}")
print(f"   Precision : {results_complete['final_performance']['precision']:.3f}")
print(f"   Recall    : {results_complete['final_performance']['recall']:.3f}")
print(f"   Final Score : {results_complete['final_performance']['final_score']:.3f}")
if results_complete['final_performance']['auc']:
    print(f"   AUC       : {results_complete['final_performance']['auc']:.3f}")

print("\n" + "="*80)
print(" ✅ PROCESSUS TERMINÉ")
print("="*80)
