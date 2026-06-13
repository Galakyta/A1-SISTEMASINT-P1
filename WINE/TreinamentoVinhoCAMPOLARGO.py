import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from ModuloAvaliador import avaliadorDemodelo
import pickle
import time
 
tinto = pd.read_csv("DATASETS/winequality-red.csv", sep=";")
branco = pd.read_csv("DATASETS/winequality-white.csv", sep=";")
 
tinto["wine_type"] = 0
branco["wine_type"] = 1
 
dados = pd.concat([tinto, branco], ignore_index=True)
 
dados_atributos = dados.drop(columns=["quality"])
target = dados["quality"]
 
le = LabelEncoder()
target_enc = le.fit_transform(target)
 
atributos_treino, atributos_teste, target_treino, target_teste = train_test_split(
    dados_atributos,
    target_enc,
    test_size=0.3,
    random_state=42,
    stratify=target_enc
)
 
scaler = StandardScaler()
atributos_treino = scaler.fit_transform(atributos_treino)
atributos_teste = scaler.transform(atributos_teste)
 
menor_classe = pd.Series(target_treino).value_counts().min()
k_viz = min(5, menor_classe - 1)
 
smoter = SMOTE(random_state=42, k_neighbors=k_viz)
atributos_treino_b, target_treino_b = smoter.fit_resample(atributos_treino, target_treino)
 
print(f"Treino apos SMOTE: {atributos_treino_b.shape[0]} amostras")
print(f"Distribuicao apos SMOTE: {np.unique(target_treino_b, return_counts=True)}\n")
 
 
grade_de_parametros_xgb = {
    "n_estimators": [int(x) for x in np.linspace(100, 1000, 10)],
    "max_depth": [3, 4, 5, 6, 7, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5, 1],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0, 0.1, 0.5, 1],
    "reg_lambda": [1, 1.5, 2, 5]
}
 
grade_de_parametros_rf = {
    "n_estimators": [int(x) for x in np.linspace(100, 1000, 10)],
    "criterion": ["gini", "entropy"],
    "max_depth": [int(x) for x in np.linspace(10, 100, 10)] + [None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"]
}
 
grade_de_parametros_hgb = {
    "max_iter": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10, None],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_samples_leaf": [10, 20, 30, 50],
    "l2_regularization": [0, 0.1, 0.5, 1.0],
    "max_leaf_nodes": [15, 31, 63, None]
}
 
 
VinhosXGB = XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss")
VinhosRF = RandomForestClassifier(random_state=42, n_jobs=-1)
VinhosHGB = HistGradientBoostingClassifier(random_state=42)
 
random_search_xgb = RandomizedSearchCV(
    VinhosXGB, param_distributions=grade_de_parametros_xgb,
    n_iter=30, cv=5, scoring="f1_weighted", n_jobs=-1, random_state=42, verbose=1
)
 
random_search_rf = RandomizedSearchCV(
    VinhosRF, param_distributions=grade_de_parametros_rf,
    n_iter=20, cv=5, scoring="f1_weighted", n_jobs=-1, random_state=42, verbose=1
)
 
random_search_hgb = RandomizedSearchCV(
    VinhosHGB, param_distributions=grade_de_parametros_hgb,
    n_iter=20, cv=5, scoring="f1_weighted", n_jobs=-1, random_state=42, verbose=1
)
print("Iniciando RandomizedSearchCV XGB...")
random_search_xgb.fit(atributos_treino_b, target_treino_b)
print(f"Melhores parametros XGB: {random_search_xgb.best_params_}")
print(f"Melhor score CV XGB (f1_weighted): {random_search_xgb.best_score_:.4f}\n")
 
print("Iniciando RandomizedSearchCV RF...")
random_search_rf.fit(atributos_treino_b, target_treino_b)
print(f"Melhores parametros RF: {random_search_rf.best_params_}")
print(f"Melhor score CV RF (f1_weighted): {random_search_rf.best_score_:.4f}\n")
 
print("Iniciando RandomizedSearchCV HGB...")
random_search_hgb.fit(atributos_treino_b, target_treino_b)
print(f"Melhores parametros HGB: {random_search_hgb.best_params_}")
print(f"Melhor score CV HGB (f1_weighted): {random_search_hgb.best_score_:.4f}\n")
 
start_time = time.time()
xgb_final = XGBClassifier(**random_search_xgb.best_params_, random_state=42, n_jobs=-1, eval_metric="mlogloss")
xgb_final.fit(atributos_treino_b, target_treino_b)
print(f"Tempo de treino XGB: {time.time() - start_time:.4f} segundos\n")
 
start_time = time.time()
rf_final = RandomForestClassifier(**random_search_rf.best_params_, random_state=42, n_jobs=-1)
rf_final.fit(atributos_treino_b, target_treino_b)
print(f"Tempo de treino RF: {time.time() - start_time:.4f} segundos\n")
 
start_time = time.time()
hgb_final = HistGradientBoostingClassifier(**random_search_hgb.best_params_, random_state=42)
hgb_final.fit(atributos_treino_b, target_treino_b)
print(f"Tempo de treino HGB: {time.time() - start_time:.4f} segundos\n")
 

target_teste_bin = (target_teste >= le.transform([7])[0]).astype(int)
 
class WrapperBinario:
    """
    tive q fazer esse wrapper que binariza as saidas do multiclass pq
    pra ser compativel com o moduloavaliador que espera uma classificacao binaria e tava dando um monte de erro na hora da chamada
    """
    def __init__(self, modelo, threshold_enc):
        self.modelo = modelo
        self.threshold_enc = threshold_enc
 
    def predict(self, X):
        pred_enc = self.modelo.predict(X)
        return (pred_enc >= self.threshold_enc).astype(int)
 
    def predict_proba(self, X):
        proba_todas = self.modelo.predict_proba(X)
        prob_bom = proba_todas[:, self.threshold_enc:].sum(axis=1)
        prob_nao_bom = 1 - prob_bom
        return np.column_stack([prob_nao_bom, prob_bom])
 
    def __repr__(self):
        return f"WrapperBinario(threshold_enc={self.threshold_enc})"
 
threshold_enc = int(le.transform([7])[0])
 
resultados = []
 
resultados.append(avaliadorDemodelo(WrapperBinario(xgb_final, threshold_enc), atributos_teste, target_teste_bin, "XGBoost"))
resultados.append(avaliadorDemodelo(WrapperBinario(rf_final, threshold_enc), atributos_teste, target_teste_bin, "Random Forest"))
resultados.append(avaliadorDemodelo(WrapperBinario(hgb_final, threshold_enc), atributos_teste, target_teste_bin, "HistGradientBoosting"))
 
df_resultados = pd.DataFrame(resultados).set_index("Modelo")
print("\n\nCOMPARATIVO FINAL")
print(df_resultados.to_string())
 
melhor = df_resultados["ROC_AUC"].idxmax()
print(f"\nModelo selecionado para implantacao: {melhor}")
 
pickle.dump(xgb_final, open("JARRO DE PICLES(pickles)/VinhoXGB.pkl", "wb"))
pickle.dump(rf_final,  open("JARRO DE PICLES(pickles)/VinhoRF.pkl",  "wb"))
pickle.dump(hgb_final, open("JARRO DE PICLES(pickles)/VinhoHGB.pkl", "wb"))
pickle.dump(scaler,    open("JARRO DE PICLES(pickles)/VinhoScaler.pkl", "wb"))
pickle.dump(le,        open("JARRO DE PICLES(pickles)/VinhoLabelEncoder.pkl", "wb"))
print("salvo salvado")


"""Treino apos SMOTE: 13895 amostras
Distribuicao apos SMOTE: (array([0, 1, 2, 3, 4, 5, 6]), array([1985, 1985, 1985, 1985, 1985, 1985, 1985]))

Iniciando RandomizedSearchCV...
Fitting 5 folds for each of 30 candidates, totalling 150 fits

Melhores parametros XGB: {'subsample': 0.7, 'reg_lambda': 2, 'reg_alpha': 0, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 10, 'learning_rate': 0.05, 'gamma': 0, 'colsample_bytree': 1.0}
Melhor score CV XGB (f1_weighted): 0.8967

Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)
Tempo de treino XGB: 2.8627 segundos


//////////////////////////////////////////////////////////////////////////////////////////////////////

                                                XGBoost Wine Quality (bom >= 7)

//////////////////////////////////////////////////////////////////////////////////////////////////////
Accrcy: 0.88
F(uma)1: 0.6953125
ROCAUC: 0.9200447879818916

Matriz de Confus�o
[[1449  118]
 [ 116  267]]

Relat�rio
              precision    recall  f1-score   support

           0       0.93      0.92      0.93      1567
           1       0.69      0.70      0.70       383

    accuracy                           0.88      1950
   macro avg       0.81      0.81      0.81      1950
weighted avg       0.88      0.88      0.88      1950"""

#esqueci de tirar o label encoder mas fazer oq, tirei agr mas saiu no terminal 