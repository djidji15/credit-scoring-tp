"""
cross_validation.py
-------------------
Comparaison approfondie de plusieurs algorithmes avec validation croisée.
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Algorithmes à comparer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# FONCTION : Score personnalisé (Accuracy + Precision) / 2
# ============================================================

def custom_score(y_true, y_pred):
    """
    Calcule le score personnalisé : (accuracy + precision) / 2
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    return (acc + prec) / 2


# ============================================================
# FONCTION PRINCIPALE : Comparaison avec validation croisée
# ============================================================

def run_classifiers_cv(X, y, classifiers, cv=10, random_state=0, verbose=True):
    """
    Compare plusieurs classifieurs avec validation croisée 10-fold.
    
    Paramètres :
    -----------
    X : array-like, features
    y : array-like, target
    classifiers : dict, dictionnaire {nom: classifieur}
    cv : int, nombre de folds
    random_state : int, graine aléatoire
    verbose : bool, afficher les résultats détaillés
    
    Retourne :
    ---------
    results_df : DataFrame, résultats de tous les classifieurs
    """
    
    if verbose:
        print("\n" + "="*80)
        print(f" COMPARAISON DE {len(classifiers)} ALGORITHMES - VALIDATION CROISÉE {cv}-FOLD")
        print("="*80)
    
    # Préparation du KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Scorers
    custom_scorer = make_scorer(custom_score)
    
    results = []
    
    for name, clf in classifiers.items():
        if verbose:
            print(f"\n🔄 Évaluation de {name}...")
        
        try:
            # Temps d'exécution
            start_time = time.time()
            
            # Cross-validation pour Accuracy
            cv_acc = cross_val_score(clf, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
            
            # Cross-validation pour AUC
            try:
                cv_auc = cross_val_score(clf, X, y, cv=kf, scoring='roc_auc', n_jobs=-1)
                auc_mean = np.mean(cv_auc)
                auc_std = np.std(cv_auc)
            except:
                auc_mean = np.nan
                auc_std = np.nan
            
            # Cross-validation pour le score personnalisé
            cv_custom = cross_val_score(clf, X, y, cv=kf, scoring=custom_scorer, n_jobs=-1)
            
            elapsed_time = time.time() - start_time
            
            # Calcul des statistiques
            acc_mean = np.mean(cv_acc)
            acc_std = np.std(cv_acc)
            custom_mean = np.mean(cv_custom)
            custom_std = np.std(cv_custom)
            
            results.append({
                'Algorithm': name,
                'Accuracy_mean': acc_mean,
                'Accuracy_std': acc_std,
                'AUC_mean': auc_mean,
                'AUC_std': auc_std,
                'Custom_Score_mean': custom_mean,
                'Custom_Score_std': custom_std,
                'Time_seconds': elapsed_time
            })
            
            if verbose:
                print(f"   ✅ Accuracy     : {acc_mean:.3f} ± {acc_std:.3f}")
                if not np.isnan(auc_mean):
                    print(f"   ✅ AUC          : {auc_mean:.3f} ± {auc_std:.3f}")
                print(f"   ✅ Custom Score : {custom_mean:.3f} ± {custom_std:.3f}")
                print(f"   ⏱️  Temps        : {elapsed_time:.2f}s")
        
        except Exception as e:
            if verbose:
                print(f"   ❌ Erreur : {str(e)}")
            results.append({
                'Algorithm': name,
                'Accuracy_mean': np.nan,
                'Accuracy_std': np.nan,
                'AUC_mean': np.nan,
                'AUC_std': np.nan,
                'Custom_Score_mean': np.nan,
                'Custom_Score_std': np.nan,
                'Time_seconds': np.nan
            })
    
    # Conversion en DataFrame
    results_df = pd.DataFrame(results)
    
    # Tri par Custom Score
    results_df = results_df.sort_values('Custom_Score_mean', ascending=False).reset_index(drop=True)
    
    if verbose:
        print("\n" + "="*80)
        print(" 📊 RÉSUMÉ DES RÉSULTATS (trié par Custom Score)")
        print("="*80)
        print(results_df.to_string(index=False))
        
        best_algo = results_df.iloc[0]
        print(f"\n🏆 MEILLEUR ALGORITHME : {best_algo['Algorithm']}")
        print(f"   Custom Score : {best_algo['Custom_Score_mean']:.3f} ± {best_algo['Custom_Score_std']:.3f}")
        print(f"   Accuracy     : {best_algo['Accuracy_mean']:.3f} ± {best_algo['Accuracy_std']:.3f}")
        if not np.isnan(best_algo['AUC_mean']):
            print(f"   AUC          : {best_algo['AUC_mean']:.3f} ± {best_algo['AUC_std']:.3f}")
    
    return results_df


# ============================================================
# FONCTION : Créer le dictionnaire de classifieurs
# ============================================================

def create_classifiers_dict(random_state=1):
    """
    Crée un dictionnaire contenant tous les classifieurs à comparer.
    
    Retourne :
    ---------
    classifiers : dict, dictionnaire {nom: classifieur}
    """
    classifiers = {
        # Arbres de décision
        'CART': DecisionTreeClassifier(random_state=random_state),
        'CART_max5': DecisionTreeClassifier(max_depth=5, random_state=random_state),
        'CART_max10': DecisionTreeClassifier(max_depth=10, random_state=random_state),
        'Decision_Stump': DecisionTreeClassifier(max_depth=1, random_state=random_state),
        
        # ID3 (simulé avec DecisionTree + entropy)
        'ID3': DecisionTreeClassifier(criterion='entropy', random_state=random_state),
        
        # MLP
        'MLP_20_10': MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=400, random_state=random_state),
        'MLP_40_20': MLPClassifier(hidden_layer_sizes=(40, 20), max_iter=400, random_state=random_state),
        'MLP_50_25': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=400, random_state=random_state),
        
        # KNN
        'KNN_k3': KNeighborsClassifier(n_neighbors=3),
        'KNN_k5': KNeighborsClassifier(n_neighbors=5),
        'KNN_k10': KNeighborsClassifier(n_neighbors=10),
        'KNN_k15': KNeighborsClassifier(n_neighbors=15),
        
        # Bagging
        'Bagging_50': BaggingClassifier(n_estimators=50, random_state=random_state, n_jobs=-1),
        'Bagging_100': BaggingClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'Bagging_200': BaggingClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        
        # AdaBoost
        'AdaBoost_50': AdaBoostClassifier(n_estimators=50, random_state=random_state, algorithm='SAMME'),
        'AdaBoost_100': AdaBoostClassifier(n_estimators=100, random_state=random_state, algorithm='SAMME'),
        'AdaBoost_200': AdaBoostClassifier(n_estimators=200, random_state=random_state, algorithm='SAMME'),
        
        # Random Forest
        'RF_50': RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1),
        'RF_100': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'RF_200': RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        
        # Gradient Boosting
        'GradientBoosting_50': GradientBoostingClassifier(n_estimators=50, random_state=random_state),
        'GradientBoosting_100': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        'GradientBoosting_200': GradientBoostingClassifier(n_estimators=200, random_state=random_state),
        
        # XGBoost
        'XGBoost_50': XGBClassifier(n_estimators=50, random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
        'XGBoost_100': XGBClassifier(n_estimators=100, random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
        'XGBoost_200': XGBClassifier(n_estimators=200, random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
    }
    
    return classifiers


# ============================================================
# FONCTION : Pipeline complet avec validation croisée
# ============================================================

def pipeline_generation_cv(
    df,
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
):
    """
    Fonction d'orchestration complète avec validation croisée pour sélectionner
    le meilleur algorithme et créer le pipeline final.
    
    Paramètres :
    -----------
    df : DataFrame, données brutes
    target_column : str, nom de la colonne cible
    test_size : float, proportion du test set
    random_state : int, graine aléatoire
    cv_folds : int, nombre de folds pour la validation croisée
    compare_preprocessing : bool, comparer différentes configurations de preprocessing
    use_scaler : bool, normaliser les données
    use_pca : bool, appliquer l'ACP
    n_components : int, nombre de composantes PCA
    auto_select_features : bool, sélectionner automatiquement les features
    n_features_to_select : int, nombre de features (si auto=False)
    tune_hyperparameters : bool, optimiser les hyperparamètres
    output_filename : str, nom du fichier pickle de sortie
    verbose : bool, afficher les logs détaillés
    
    Retourne :
    ---------
    final_pipeline : Pipeline sklearn entraîné et optimisé
    results : dict, résultats de toutes les étapes
    """
    
    from utils import (
        train_test_classifiers_with_pca,
        select_best_features,
        find_best_hyperparameters,
        create_pipeline,
        save_pipeline
    )
    from sklearn.model_selection import train_test_split
    
    results = {}
    
    if verbose:
        print("\n" + "="*80)
        print(" ORCHESTRATION COMPLÈTE AVEC VALIDATION CROISÉE")
        print("="*80)
    
    # ===== ÉTAPE 1 : Préparation des données =====
    if verbose:
        print("\n[1/6] 📊 Préparation des données...")
    
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if verbose:
        print(f"   ✅ Train: {X_train.shape} | Test: {X_test.shape}")
    
    # ===== ÉTAPE 2 : Comparaison sur données brutes =====
    if compare_preprocessing:
        if verbose:
            print("\n[2/6] 🔬 Comparaison sur données BRUTES...")
        
        classifiers = create_classifiers_dict(random_state)
        results_raw = run_classifiers_cv(X_train, y_train, classifiers, cv=cv_folds, verbose=verbose)
        results['raw_data'] = results_raw
    
    # ===== ÉTAPE 3 : Comparaison sur données normalisées =====
    if verbose:
        print("\n[3/6] 📏 Comparaison sur données NORMALISÉES...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifiers = create_classifiers_dict(random_state)
    results_norm = run_classifiers_cv(X_train_scaled, y_train, classifiers, cv=cv_folds, verbose=verbose)
    results['normalized_data'] = results_norm
    
    # ===== ÉTAPE 4 : Comparaison sur données avec PCA =====
    if use_pca:
        if verbose:
            print("\n[4/6] 🧮 Comparaison sur données NORMALISÉES + PCA...")
        
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        X_train_final = np.concatenate([X_train_scaled, X_train_pca], axis=1)
        X_test_final = np.concatenate([X_test_scaled, X_test_pca], axis=1)
        
        classifiers = create_classifiers_dict(random_state)
        results_pca = run_classifiers_cv(X_train_final, y_train, classifiers, cv=cv_folds, verbose=verbose)
        results['pca_data'] = results_pca
        
        # Utiliser les données PCA pour la suite
        X_train_processed = X_train_final
        X_test_processed = X_test_final
    else:
        X_train_processed = X_train_scaled
        X_test_processed = X_test_scaled
        results_pca = results_norm
    
    # ===== ÉTAPE 5 : Sélection du meilleur algorithme =====
    if verbose:
        print("\n[5/6] 🏆 Sélection du meilleur algorithme...")
    
    best_algo_row = results_pca.iloc[0]
    best_algo_name = best_algo_row['Algorithm']
    best_algo_score = best_algo_row['Custom_Score_mean']
    
    if verbose:
        print(f"   ✅ Meilleur algorithme : {best_algo_name}")
        print(f"   ✅ Custom Score        : {best_algo_score:.3f} ± {best_algo_row['Custom_Score_std']:.3f}")
    
    # Déterminer le type de classifieur et ses paramètres
    classifier_type, classifier_params = extract_classifier_info(best_algo_name)
    
    results['best_algorithm'] = {
        'name': best_algo_name,
        'type': classifier_type,
        'params': classifier_params,
        'custom_score_mean': best_algo_score,
        'custom_score_std': best_algo_row['Custom_Score_std']
    }
    
    # ===== ÉTAPE 6 : Sélection de features et création du pipeline final =====
    if verbose:
        print(f"\n[6/6] 🏗️  Création du pipeline final avec {best_algo_name}...")
    
    # Sélection de features
    if auto_select_features:
        original_features = df.drop(columns=[target_column]).columns.tolist()
        if use_pca:
            feature_names = original_features + [f"PCA{i+1}" for i in range(n_components)]
        else:
            feature_names = original_features
        
        best_k, sorted_indices, importances, scores_features = select_best_features(
            X_train_processed, y_train, X_test_processed, y_test,
            feature_names=feature_names,
            classifier_type=classifier_type,
            classifier_params=classifier_params,
            verbose=verbose
        )
        
        n_features_to_select = best_k
        results['feature_selection'] = {
            'best_k': best_k,
            'selected_features': [feature_names[i] for i in sorted_indices[:best_k]]
        }
    
    # Optimisation des hyperparamètres si demandé
    if tune_hyperparameters:
        if verbose:
            print(f"\n⚙️  Optimisation des hyperparamètres pour {classifier_type}...")
        
        if n_features_to_select and auto_select_features:
            X_train_optim = X_train_processed[:, sorted_indices[:n_features_to_select]]
        else:
            X_train_optim = X_train_processed
        
        best_params, _ = find_best_hyperparameters(
            X_train_optim, y_train,
            classifier_type=classifier_type,
            param_grid=None,
            cv=5,
            verbose=verbose
        )
        
        classifier_params = best_params
        results['optimized_hyperparameters'] = best_params
    
    # Création du pipeline final
    final_pipeline = create_pipeline(
        use_scaler=use_scaler,
        use_pca=use_pca,
        n_components=n_components,
        n_features_to_select=n_features_to_select,
        classifier_type=classifier_type,
        classifier_params=classifier_params,
        verbose=verbose
    )
    
    # Entraînement du pipeline final
    if verbose:
        print("\n🔄 Entraînement du pipeline final...")
    
    final_pipeline.fit(X_train, y_train)
    
    # Évaluation finale
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    
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
        print("\n" + "="*80)
        print(" PERFORMANCES FINALES DU PIPELINE")
        print("="*80)
        print(f"Algorithme   : {best_algo_name}")
        print(f"Accuracy     : {acc:.3f}")
        print(f"Precision    : {prec:.3f}")
        print(f"Recall       : {rec:.3f}")
        print(f"Final Score  : {final_score:.3f}")
        if auc:
            print(f"AUC          : {auc:.3f}")
    
    # Sauvegarde
    save_pipeline(final_pipeline, output_filename, verbose=verbose)
    
    if verbose:
        print("\n" + "="*80)
        print(" ✅ ORCHESTRATION AVEC VALIDATION CROISÉE TERMINÉE")
        print("="*80)
    
    return final_pipeline, results


# ============================================================
# FONCTION UTILITAIRE : Extraction d'informations du classifieur
# ============================================================

def extract_classifier_info(algo_name):
    """
    Extrait le type et les paramètres d'un classifieur à partir de son nom.
    
    Paramètres :
    -----------
    algo_name : str, nom de l'algorithme
    
    Retourne :
    ---------
    classifier_type : str, type du classifieur
    classifier_params : dict, paramètres du classifieur
    """
    if 'CART' in algo_name:
        classifier_type = 'CART'
        if 'max5' in algo_name:
            classifier_params = {'max_depth': 5}
        elif 'max10' in algo_name:
            classifier_params = {'max_depth': 10}
        else:
            classifier_params = {}
    
    elif 'ID3' in algo_name:
        classifier_type = 'CART'
        classifier_params = {'criterion': 'entropy'}
    
    elif 'Decision_Stump' in algo_name:
        classifier_type = 'CART'
        classifier_params = {'max_depth': 1}
    
    elif 'MLP' in algo_name:
        classifier_type = 'MLP'
        if '20_10' in algo_name:
            classifier_params = {'hidden_layer_sizes': (20, 10)}
        elif '40_20' in algo_name:
            classifier_params = {'hidden_layer_sizes': (40, 20)}
        elif '50_25' in algo_name:
            classifier_params = {'hidden_layer_sizes': (50, 25)}
        else:
            classifier_params = {'hidden_layer_sizes': (40, 20)}
    
    elif 'KNN' in algo_name:
        classifier_type = 'KNN'
        if 'k3' in algo_name:
            classifier_params = {'n_neighbors': 3}
        elif 'k5' in algo_name:
            classifier_params = {'n_neighbors': 5}
        elif 'k10' in algo_name:
            classifier_params = {'n_neighbors': 10}
        elif 'k15' in algo_name:
            classifier_params = {'n_neighbors': 15}
        else:
            classifier_params = {'n_neighbors': 5}
    
    elif 'RF' in algo_name or 'Random' in algo_name:
        classifier_type = 'RF'
        if '50' in algo_name:
            classifier_params = {'n_estimators': 50}
        elif '100' in algo_name:
            classifier_params = {'n_estimators': 100}
        elif '200' in algo_name:
            classifier_params = {'n_estimators': 200}
        else:
            classifier_params = {'n_estimators': 200}
    
    elif 'XGBoost' in algo_name:
        classifier_type = 'XGBoost'
        if '50' in algo_name:
            classifier_params = {'n_estimators': 50}
        elif '100' in algo_name:
            classifier_params = {'n_estimators': 100}
        elif '200' in algo_name:
            classifier_params = {'n_estimators': 200}
        else:
            classifier_params = {'n_estimators': 200}
    
    elif 'Bagging' in algo_name:
        classifier_type = 'Bagging'
        if '50' in algo_name:
            classifier_params = {'n_estimators': 50}
        elif '100' in algo_name:
            classifier_params = {'n_estimators': 100}
        elif '200' in algo_name:
            classifier_params = {'n_estimators': 200}
        else:
            classifier_params = {'n_estimators': 200}
    
    elif 'AdaBoost' in algo_name:
        classifier_type = 'AdaBoost'
        if '50' in algo_name:
            classifier_params = {'n_estimators': 50}
        elif '100' in algo_name:
            classifier_params = {'n_estimators': 100}
        elif '200' in algo_name:
            classifier_params = {'n_estimators': 200}
        else:
            classifier_params = {'n_estimators': 200}
    
    elif 'GradientBoosting' in algo_name:
        classifier_type = 'GradientBoosting'
        if '50' in algo_name:
            classifier_params = {'n_estimators': 50}
        elif '100' in algo_name:
            classifier_params = {'n_estimators': 100}
        elif '200' in algo_name:
            classifier_params = {'n_estimators': 200}
        else:
            classifier_params = {'n_estimators': 200}
    
    else:
        classifier_type = 'MLP'
        classifier_params = {'hidden_layer_sizes': (40, 20)}
    
    return classifier_type, classifier_params
