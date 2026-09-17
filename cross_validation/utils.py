"""
utils.py
--------
Module contenant toutes les fonctions nécessaires pour le pipeline de credit scoring.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, recall_score, precision_score,
    roc_curve, roc_auc_score, make_scorer
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier


# ============================================================
# CLASSES PERSONNALISÉES
# ============================================================

class PCAWithConcatenation(BaseEstimator, TransformerMixin):
    """
    Transformateur personnalisé qui applique PCA et concatène 
    les composantes principales aux données originales.
    """
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=1)
    
    def fit(self, X, y=None):
        self.pca.fit(X)
        return self
    
    def transform(self, X):
        X_pca = self.pca.transform(X)
        X_concat = np.concatenate([X, X_pca], axis=1)
        return X_concat


class RFFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Sélectionneur de features basé sur l'importance Random Forest.
    Compatible avec pickle.
    """
    def __init__(self, k=12, n_estimators=100):
        self.k = k
        self.n_estimators = n_estimators
        self.selected_indices_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=1, 
            n_jobs=-1
        )
        rf.fit(X, y)
        self.feature_importances_ = rf.feature_importances_
        self.selected_indices_ = np.argsort(self.feature_importances_)[::-1][:self.k]
        return self
    
    def transform(self, X):
        if self.selected_indices_ is None:
            raise ValueError("Le sélecteur doit être fit() avant transform()")
        return X[:, self.selected_indices_]


# ============================================================
# FONCTION 1 : Apprentissage et test des algorithmes supervisés
# ============================================================

def train_test_classifiers(X_train, y_train, X_test, y_test, classifiers=None, verbose=True):
    """
    Entraîne et évalue plusieurs classifieurs sur les données brutes.
    
    Paramètres :
    -----------
    X_train, y_train : données d'entraînement
    X_test, y_test : données de test
    classifiers : dict, classifieurs à tester (None = défaut)
    verbose : bool, afficher les résultats détaillés
    
    Retourne :
    ---------
    scores : dict, métriques pour chaque classifieur
    """
    if classifiers is None:
        classifiers = {
            "CART": DecisionTreeClassifier(random_state=1),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "MLP": MLPClassifier(hidden_layer_sizes=(40, 20), max_iter=400, random_state=1)
        }
    
    scores = {}
    
    for name, model in classifiers.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f" Entraînement : {name}")
            print('='*60)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        final_score = (acc + prec) / 2
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = None
        
        scores[name] = {
            "model": model,
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "final_score": final_score,
            "auc": auc
        }
        
        if verbose:
            print(f"Accuracy  : {acc:.3f}")
            print(f"Precision : {prec:.3f}")
            print(f"Recall    : {rec:.3f}")
            print(f"Final Score : {final_score:.3f}")
            if auc:
                print(f"AUC       : {auc:.3f}")
    
    best_model_name = max(scores, key=lambda x: scores[x]["final_score"])
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🏆 Meilleur modèle : {best_model_name} (Final Score = {scores[best_model_name]['final_score']:.3f})")
        print('='*60)
    
    return scores, best_model_name


# ============================================================
# FONCTION 2 : Apprentissage sur données normalisées
# ============================================================

def train_test_classifiers_normalized(X_train, y_train, X_test, y_test, classifiers=None, verbose=True):
    """
    Entraîne et évalue plusieurs classifieurs sur les données normalisées.
    
    Paramètres :
    -----------
    X_train, y_train : données d'entraînement
    X_test, y_test : données de test
    classifiers : dict, classifieurs à tester
    verbose : bool, afficher les résultats
    
    Retourne :
    ---------
    scores : dict, métriques pour chaque classifieur
    scaler : StandardScaler ajusté
    """
    if verbose:
        print("\n📊 Normalisation des données...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        print("✅ Normalisation terminée\n")
    
    scores, best_model_name = train_test_classifiers(
        X_train_scaled, y_train, X_test_scaled, y_test, 
        classifiers, verbose
    )
    
    return scores, best_model_name, scaler


# ============================================================
# FONCTION 3 : Apprentissage avec nouvelles variables (PCA)
# ============================================================

def train_test_classifiers_with_pca(X_train, y_train, X_test, y_test, 
                                     n_components=3, classifiers=None, verbose=True):
    """
    Entraîne et évalue plusieurs classifieurs sur les données normalisées + PCA.
    
    Paramètres :
    -----------
    X_train, y_train : données d'entraînement
    X_test, y_test : données de test
    n_components : int, nombre de composantes PCA
    classifiers : dict, classifieurs à tester
    verbose : bool, afficher les résultats
    
    Retourne :
    ---------
    scores : dict, métriques pour chaque classifieur
    scaler : StandardScaler ajusté
    pca : PCA ajusté
    """
    if verbose:
        print("\n📊 Normalisation + ACP...")
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ACP
    pca = PCA(n_components=n_components, random_state=1)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Concaténation
    X_train_final = np.concatenate([X_train_scaled, X_train_pca], axis=1)
    X_test_final = np.concatenate([X_test_scaled, X_test_pca], axis=1)
    
    if verbose:
        print(f"✅ ACP effectuée : {n_components} composantes")
        print(f"   Variance expliquée : {np.sum(pca.explained_variance_ratio_):.3f}")
        print(f"   Dimensions finales : {X_train_final.shape}\n")
    
    scores, best_model_name = train_test_classifiers(
        X_train_final, y_train, X_test_final, y_test, 
        classifiers, verbose
    )
    
    return scores, best_model_name, scaler, pca


# ============================================================
# FONCTION 4 : Sélection des variables pertinentes
# ============================================================

def select_best_features(X_train, y_train, X_test, y_test, 
                        feature_names=None, classifier_type='MLP',
                        classifier_params=None, verbose=True):
    """
    Sélectionne le nombre optimal de variables en testant progressivement.
    
    Paramètres :
    -----------
    X_train, y_train : données d'entraînement
    X_test, y_test : données de test
    feature_names : list, noms des features
    classifier_type : str, type de classifieur ('MLP', 'CART', 'KNN')
    classifier_params : dict, paramètres du classifieur
    verbose : bool, afficher les résultats
    
    Retourne :
    ---------
    best_k : int, nombre optimal de features
    sorted_indices : array, indices triés par importance
    importances : array, importances des features
    """
    if verbose:
        print("\n🔍 Calcul de l'importance des variables...")
    
    # Calcul de l'importance avec Random Forest
    rf = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    if verbose and feature_names is not None:
        print("\n📊 Variables triées par importance :")
        for i, idx in enumerate(sorted_indices[:10], 1):
            print(f"  {i}. {feature_names[idx]} : {importances[idx]:.4f}")
    
    # Test progressif du nombre de variables
    if verbose:
        print("\n🔄 Test du nombre optimal de variables...")
    
    scores = np.zeros(X_train.shape[1])
    
    if classifier_type == 'MLP':
        if classifier_params is None:
            classifier_params = {'hidden_layer_sizes': (40, 20), 'max_iter': 400}
        clf = MLPClassifier(random_state=1, **classifier_params)
    elif classifier_type == 'CART':
        clf = DecisionTreeClassifier(random_state=1, **(classifier_params or {}))
    elif classifier_type == 'KNN':
        clf = KNeighborsClassifier(**(classifier_params or {}))
    elif classifier_type == 'RF':
        clf = RandomForestClassifier(random_state=1, n_jobs=-1, **classifier_params)
    elif classifier_type == 'XGBoost':
        clf = XGBClassifier(random_state=1,use_label_encoder=False,eval_metric='logloss',**classifier_params)
    elif classifier_type == 'Bagging':
        clf = BaggingClassifier(random_state=1,n_jobs=-1,**classifier_params)
    elif classifier_type == 'AdaBoost':
        clf = AdaBoostClassifier(random_state=1,algorithm='SAMME',**classifier_params)
    elif classifier_type == 'GradientBoosting':
        clf = GradientBoostingClassifier(random_state=1,**classifier_params)
    else:
        raise ValueError(f"Type de classifieur inconnu : {classifier_type}")
    
    for k in range(1, X_train.shape[1] + 1):
        X_train_k = X_train[:, sorted_indices[:k]]
        X_test_k = X_test[:, sorted_indices[:k]]
        
        clf.fit(X_train_k, y_train)
        y_pred = clf.predict(X_test_k)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        scores[k-1] = (acc + prec) / 2
    
    best_k = np.argmax(scores) + 1
    
    if verbose:
        print(f"✅ Nombre optimal de variables : {best_k} (Final Score = {scores[best_k-1]:.3f})")
    
    return best_k, sorted_indices, importances, scores


# ============================================================
# FONCTION 5 : Recherche des meilleurs hyperparamètres
# ============================================================

def find_best_hyperparameters(X_train, y_train, classifier_type='MLP', 
                             param_grid=None, cv=5, verbose=True):
    """
    Recherche les meilleurs hyperparamètres via GridSearchCV.
    
    Paramètres :
    -----------
    X_train, y_train : données d'entraînement
    classifier_type : str, type de classifieur
    param_grid : dict, grille de paramètres à tester
    cv : int, nombre de folds pour la validation croisée
    verbose : bool, afficher les résultats
    
    Retourne :
    ---------
    best_params : dict, meilleurs paramètres trouvés
    best_model : modèle optimisé
    """
    if verbose:
        print(f"\n🔍 GridSearchCV pour {classifier_type}...")
    
    # Score personnalisé
    def custom_score(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        return (acc + prec) / 2
    
    scorer = make_scorer(custom_score)
    
    # Définition du modèle et grille par défaut
    if classifier_type == 'MLP':
        base_model = MLPClassifier(max_iter=400, random_state=1)
        if param_grid is None:
            param_grid = {
                'hidden_layer_sizes': [(30, 15), (40, 20), (50, 25)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
    elif classifier_type == 'CART':
        base_model = DecisionTreeClassifier(random_state=1)
        if param_grid is None:
            param_grid = {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
    elif classifier_type == 'KNN':
        base_model = KNeighborsClassifier()
        if param_grid is None:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
    elif classifier_type == 'RF':
        base_model = RandomForestClassifier(random_state=1, n_jobs=-1)
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
    
    elif classifier_type == 'XGBoost':
        base_model = XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss')
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
    
    elif classifier_type == 'Bagging':
        base_model = BaggingClassifier(random_state=1, n_jobs=-1)
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_samples': [0.5, 0.7, 1.0]
            }
    
    elif classifier_type == 'AdaBoost':
        base_model = AdaBoostClassifier(random_state=1, algorithm='SAMME')
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.5, 1.0, 1.5]
            }
    
    elif classifier_type == 'GradientBoosting':
        base_model = GradientBoostingClassifier(random_state=1)
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
    else:
        raise ValueError(f"Type de classifieur inconnu : {classifier_type}")
    
    # GridSearch
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    grid.fit(X_train, y_train)
    
    if verbose:
        print(f"✅ Meilleurs paramètres : {grid.best_params_}")
        print(f"   Score CV : {grid.best_score_:.3f}")
    
    return grid.best_params_, grid.best_estimator_


# ============================================================
# FONCTION 6 : Création du pipeline
# ============================================================

def create_pipeline(use_scaler=True, use_pca=True, n_components=3,
                   n_features_to_select=12, classifier_type='MLP',
                   classifier_params=None, verbose=True):
    """
    Crée un pipeline complet pour le credit scoring.
    
    Paramètres :
    -----------
    use_scaler : bool, appliquer la normalisation
    use_pca : bool, appliquer l'ACP avec concaténation
    n_components : int, nombre de composantes PCA
    n_features_to_select : int, nombre de features à sélectionner
    classifier_type : str, type de classifieur
    classifier_params : dict, paramètres du classifieur
    verbose : bool, afficher les étapes
    
    Retourne :
    ---------
    pipeline : Pipeline sklearn
    """
    steps = []
    
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    
    if use_pca:
        steps.append(('pca_concat', PCAWithConcatenation(n_components=n_components)))
    
    if n_features_to_select is not None:
        steps.append(('feature_selection', 
                     RFFeatureSelector(k=n_features_to_select, n_estimators=100)))
    
    # Classifieur
    if classifier_params is None:
        classifier_params = {}
    
    if classifier_type == 'MLP':
        clf = MLPClassifier(
            hidden_layer_sizes=classifier_params.get('hidden_layer_sizes', (40, 20)),
            activation=classifier_params.get('activation', 'relu'),
            alpha=classifier_params.get('alpha', 0.001),
            learning_rate_init=classifier_params.get('learning_rate_init', 0.001),
            max_iter=400,
            random_state=1
        )
    elif classifier_type == 'CART':
        clf = DecisionTreeClassifier(random_state=1, **classifier_params)
    elif classifier_type == 'KNN':
        clf = KNeighborsClassifier(**classifier_params)
    elif classifier_type == 'RF':
        clf = RandomForestClassifier(random_state=1, n_jobs=-1, **classifier_params)
    elif classifier_type == 'XGBoost':
        clf = XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss', **classifier_params)
    elif classifier_type == 'Bagging':
        clf = BaggingClassifier(random_state=1, n_jobs=-1, **classifier_params)
    elif classifier_type == 'AdaBoost':
        clf = AdaBoostClassifier(random_state=1, algorithm='SAMME', **classifier_params)
    elif classifier_type == 'GradientBoosting':
        clf = GradientBoostingClassifier(random_state=1, **classifier_params)
    else:
        raise ValueError(f"Type de classifieur inconnu : {classifier_type}")
    
    steps.append(('classifier', clf))
    
    pipeline = Pipeline(steps)
    
    if verbose:
        print(f"\n✅ Pipeline créé avec {len(steps)} étapes :")
        for name, _ in steps:
            print(f"   - {name}")
    
    return pipeline


# ============================================================
# FONCTION 7 : Sauvegarde et chargement
# ============================================================

def save_pipeline(pipeline, filename='credit_scoring_pipeline.pkl', verbose=True):
    """Sauvegarde le pipeline dans un fichier pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)
    if verbose:
        print(f"💾 Pipeline sauvegardé : {filename}")


def load_pipeline(filename='credit_scoring_pipeline.pkl', verbose=True):
    """Charge un pipeline depuis un fichier pickle."""
    with open(filename, 'rb') as f:
        pipeline = pickle.load(f)
    if verbose:
        print(f"📂 Pipeline chargé : {filename}")
    return pipeline


# ============================================================
# FONCTION PRINCIPALE : ORCHESTRATION COMPLÈTE
# ============================================================

def pipeline_generation_train_test_split(
    df, 
    target_column='Status',
    test_size=0.5,
    random_state=1,
    use_scaler=True,
    use_pca=True,
    n_components=3,
    auto_select_features=True,
    n_features_to_select=None,
    classifier_type='MLP',
    tune_hyperparameters=True,
    param_grid=None,
    output_filename='credit_scoring_pipeline.pkl',
    verbose=True
):
    """
    Fonction principale d'orchestration complète du pipeline de credit scoring.
    
    Paramètres :
    -----------
    df : DataFrame, données brutes
    target_column : str, nom de la colonne cible
    test_size : float, proportion du test set
    random_state : int, graine aléatoire
    use_scaler : bool, normaliser les données
    use_pca : bool, appliquer l'ACP
    n_components : int, nombre de composantes PCA
    auto_select_features : bool, sélectionner automatiquement le nombre optimal de features
    n_features_to_select : int, nombre de features (si auto_select_features=False)
    classifier_type : str, type de classifieur ('MLP', 'CART', 'KNN')
    tune_hyperparameters : bool, optimiser les hyperparamètres
    param_grid : dict, grille de paramètres pour GridSearch
    output_filename : str, nom du fichier pickle de sortie
    verbose : bool, afficher les logs détaillés
    
    Retourne :
    ---------
    pipeline : Pipeline sklearn entraîné et optimisé
    results : dict, résultats de toutes les étapes
    """
    
    results = {}
    
    if verbose:
        print("\n" + "="*70)
        print(" ORCHESTRATION COMPLÈTE DU PIPELINE DE CREDIT SCORING")
        print("="*70)
    
    # ===== ÉTAPE 1 : Préparation des données =====
    if verbose:
        print("\n[1/7] 📊 Préparation des données...")
    
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if verbose:
        print(f"   ✅ Train: {X_train.shape} | Test: {X_test.shape}")
        print(f"   ✅ Classe positive: {np.sum(y==1)/len(y)*100:.1f}%")
    
    results['data_shape'] = {'train': X_train.shape, 'test': X_test.shape}
    
    # ===== ÉTAPE 2 : Test sur données brutes =====
    if verbose:
        print("\n[2/7] 🔬 Test sur données brutes...")
    
    scores_raw, best_raw = train_test_classifiers(
        X_train, y_train, X_test, y_test, verbose=verbose
    )
    results['scores_raw'] = scores_raw
    
    # ===== ÉTAPE 3 : Test sur données normalisées =====
    if use_scaler:
        if verbose:
            print("\n[3/7] 📏 Test sur données normalisées...")
        
        scores_norm, best_norm, scaler = train_test_classifiers_normalized(
            X_train, y_train, X_test, y_test, verbose=verbose
        )
        results['scores_normalized'] = scores_norm
    
    # ===== ÉTAPE 4 : Test avec PCA =====
    if use_pca:
        if verbose:
            print("\n[4/7] 🧮 Test avec ACP...")
        
        scores_pca, best_pca, scaler, pca = train_test_classifiers_with_pca(
            X_train, y_train, X_test, y_test, 
            n_components=n_components, verbose=verbose
        )
        results['scores_pca'] = scores_pca
        
        # Préparation des données finales pour les étapes suivantes
        X_train_processed = scaler.transform(X_train)
        X_test_processed = scaler.transform(X_test)
        
        X_train_pca = pca.transform(X_train_processed)
        X_test_pca = pca.transform(X_test_processed)
        
        X_train_final = np.concatenate([X_train_processed, X_train_pca], axis=1)
        X_test_final = np.concatenate([X_test_processed, X_test_pca], axis=1)
    elif use_scaler:
        X_train_final = scaler.transform(X_train)
        X_test_final = scaler.transform(X_test)
    else:
        X_train_final = X_train
        X_test_final = X_test
    
    # ===== ÉTAPE 5 : Sélection de features =====
    if auto_select_features:
        if verbose:
            print("\n[5/7] 🎯 Sélection automatique des features...")
        
        # Noms des features
        original_features = df.drop(columns=[target_column]).columns.tolist()
        if use_pca:
            feature_names = original_features + [f"PCA{i+1}" for i in range(n_components)]
        else:
            feature_names = original_features
        
        best_k, sorted_indices, importances, scores_features = select_best_features(
            X_train_final, y_train, X_test_final, y_test,
            feature_names=feature_names,
            classifier_type=classifier_type,
            verbose=verbose
        )
        
        n_features_to_select = best_k
        results['feature_selection'] = {
            'best_k': best_k,
            'sorted_indices': sorted_indices,
            'importances': importances,
            'scores': scores_features
        }
    else:
        if verbose and n_features_to_select:
            print(f"\n[5/7] 🎯 Sélection de {n_features_to_select} features (manuel)...")
    
    # ===== ÉTAPE 6 : Optimisation des hyperparamètres =====
    classifier_params = None
    
    if tune_hyperparameters:
        if verbose:
            print("\n[6/7] ⚙️  Optimisation des hyperparamètres...")
        
        # Sélection des features si nécessaire
        if n_features_to_select and auto_select_features:
            X_train_optim = X_train_final[:, sorted_indices[:n_features_to_select]]
        else:
            X_train_optim = X_train_final
        
        best_params, best_model = find_best_hyperparameters(
            X_train_optim, y_train,
            classifier_type=classifier_type,
            param_grid=param_grid,
            verbose=verbose
        )
        
        classifier_params = best_params
        results['best_hyperparameters'] = best_params
    
    # ===== ÉTAPE 7 : Création et entraînement du pipeline final =====
    if verbose:
        print("\n[7/7] 🏗️  Création du pipeline final...")
    
    final_pipeline = create_pipeline(
        use_scaler=use_scaler,
        use_pca=use_pca,
        n_components=n_components,
        n_features_to_select=n_features_to_select,
        classifier_type=classifier_type,
        classifier_params=classifier_params,
        verbose=verbose
    )
    
    if verbose:
        print("\n🔄 Entraînement du pipeline final...")
    
    final_pipeline.fit(X_train, y_train)
    
    # Évaluation finale
    y_pred = final_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    final_score = (acc + prec) / 2
    
    if hasattr(final_pipeline.named_steps['classifier'], 'predict_proba'):
        y_proba = final_pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None
    
    results['final_performance'] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'final_score': final_score,
        'auc': auc
    }
    
    if verbose:
        print("\n" + "="*70)
        print(" PERFORMANCES FINALES DU PIPELINE")
        print("="*70)
        print(f"Accuracy  : {acc:.3f}")
        print(f"Precision : {prec:.3f}")
        print(f"Recall    : {rec:.3f}")
        print(f"Final Score : {final_score:.3f}")
        if auc:
            print(f"AUC       : {auc:.3f}")
    
    # ===== Sauvegarde =====
    save_pipeline(final_pipeline, output_filename, verbose=verbose)
    
    if verbose:
        print("\n" + "="*70)
        print(" ✅ ORCHESTRATION TERMINÉE AVEC SUCCÈS")
        print("="*70)
    
    return final_pipeline, results
