# credit-scoring-tp
Prédiction du scoring de crédit avec Python (CART, k-NN, MLP) : feature engineering, PCA, validation croisée et déploiement via API.
# TP Data Mining — Scoring de crédit

**Travail réalisé en groupe par 3 membres :**
- Redjah Nacer
- Houache Youcef
- Yennek Aldjia

## Présentation générale

Ce projet a pour objectif de construire un modèle d'apprentissage automatique capable de prédire si un client remboursera ou non son crédit (`Status` = 1 ou 0), à partir de ses informations personnelles et financières (ancienneté, statut du logement, âge, situation familiale, emploi, revenus, dettes, montant du prêt, etc.).

Le TP est organisé en deux grandes parties :

1. **Feature engineering et classification** — exploration des données, préparation, comparaison de plusieurs modèles de classification.
2. **Données hétérogènes** — traitement d'un second jeu de données mêlant variables catégorielles et numériques.

## Démarche suivie

- Exploration et nettoyage du dataset (4375 clients, 13 variables explicatives, 1 variable cible)
- Séparation des données en apprentissage / test (50/50, stratifiée)
- Comparaison de trois classifieurs : **CART** (arbre de décision), **k-NN**, **MLP**
- Normalisation des données (StandardScaler) pour améliorer les performances des modèles sensibles à l'échelle
- Réduction de dimension par **ACP (PCA)** et enrichissement du jeu de données
- Analyse de l'importance des variables (Random Forest)
- Validation croisée pour comparer et sélectionner le meilleur modèle
- Déploiement du modèle final sous forme d'**API**

**Résultat :** le modèle **MLP**, entraîné sur les données normalisées et enrichies par les composantes ACP, obtient les meilleures performances (score ≈ 0,815 — AUC ≈ 0,821).

## Contenu du dossier

| Fichier / dossier | Description |
|---|---|
| `tp1.ipynb` | Notebook principal contenant tout le code et les commentaires pour les questions 1 à 9 de la partie 1, ainsi que les 2 questions de la partie 2 |
| `credit_scoring.csv` | Jeu de données principal (scoring de crédit) |
| `credit.data` | Second jeu de données (variables hétérogènes) |
| `credit_scoring_pipeline.pkl` | Pipeline du modèle final sauvegardé |
| `image.png` | Illustration / graphique associé au notebook |
| `utils_2/` | Code de l'API permettant d'exposer le modèle (`api.py`, `main.py`, `utils.py`, `test_api.py`) |
| `cross_validation/` | Scripts et résultats de la validation croisée comparant les différents modèles (`cross_validation.py`, `main_cv.py`, `utils.py`, `comparison_results.csv`, `result_cross_validation.xlsx`) |

## Comment utiliser ce projet

1. Ouvrir `tp1.ipynb` pour suivre la démarche complète d'analyse et de modélisation.
2. Consulter le dossier `cross_validation/` pour voir la comparaison détaillée des modèles.
3. Consulter le dossier `utils_2/` pour tester l'API de scoring (lancer `main.py`, puis tester avec `test_api.py`).
