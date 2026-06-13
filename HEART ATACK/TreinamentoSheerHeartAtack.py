import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ModuloAvaliador import avaliadorDemodelo
import pickle

dados = pd.read_csv("DATASETS/heart_failure_clinical_records_dataset.csv", sep = ",")

dados_atributos = dados.drop(columns=["DEATH_EVENT"])

target = dados["DEATH_EVENT"]

#sempre separar o target 

#por ser uma random forest até dispensaria algumas formas de tratamento mas vou fazer mesmo assim
#nn vo subsituir nada por null pq eu sei q n tem
#smote pra ja balancear tudo
#dai eu tmbm pensei agora que por ser apenas numericos pode ser mt bom a vector machine mesmo que eu n goste dela, ent vo testar as duas

smoter = SMOTE(random_state=42)

atributos_treino, atributos_teste, target_treino, target_teste = train_test_split(
    dados_atributos,
    target,
    test_size=0.3,
    random_state=42,
    stratify=target
)

scaler = StandardScaler()

atributos_treino = scaler.fit_transform(atributos_treino)
atributos_teste = scaler.transform(atributos_teste)

atributos_treino_b, target_treino_b =  smoter.fit_resample(atributos_treino, target_treino)


grade_de_parametros_rf = {
    "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [int(x) for x in np.linspace(start=2, stop=10, num=2)],
    "max_depth": [int(x) for x in np.linspace(start=10, stop=100, num=20)],
    "max_features":["sqrt", "log2"]
}

grade_de_parametros_SVC = {
    "C": [0.01, 0.1, 1, 10, 100, 1000],
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "gamma": ["scale", "auto", 0.1, 0.01, 0.001, 0.0001],
    "degree": [2, 3, 4, 5],
    "coef0": [0.0, 0.1, 0.5, 1.0]
}


AraucariasRF = RandomForestClassifier(random_state=42, n_jobs=-1)
AraucariasSVC = SVC(
    random_state=42,
    probability=False
)

random_search = RandomizedSearchCV( # eu particularmente gostei batante do randomized, eu testei usar o grid na hora de fazer os exercicios e enquanto eu tava estuando, mas simplesmente prefiro
    # o randomized q é  até mais agradavel de paramentar pra mim 
    AraucariasRF,
    param_distributions=grade_de_parametros_rf,
    n_iter=20,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search_svc = RandomizedSearchCV(
    AraucariasSVC,
    param_distributions=grade_de_parametros_SVC,
    n_iter=30,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    random_state=42
)


random_search.fit(atributos_treino_b, target_treino_b)

random_search_svc.fit(atributos_treino_b, target_treino_b)


print("Melhores parâmetros rf:", random_search.best_params_)
print("Melhor score CV rf:", random_search.best_score_)

print("Melhores parâmetros svc:", random_search_svc.best_params_)
print("Melhor score CV svc:", random_search_svc.best_score_)


melhores_parametros = random_search.best_params_
melhores_parametros_svc = random_search_svc.best_params_

start_time = time.time()
rf_final = RandomForestClassifier(**melhores_parametros, random_state=42, n_jobs=-1)
rf_final.fit(atributos_treino_b, target_treino_b)
tempo_treino = time.time() - start_time
print(f"Tempo de treino RF: {tempo_treino:.4f} segundos")

start_time = time.time()
svc_final = SVC(**melhores_parametros_svc, random_state=42, probability=True)
svc_final.fit(atributos_treino_b, target_treino_b)
tempo_treino = time.time() - start_time
print(f"Tempo de treino SVC: {tempo_treino:.4f} segundos")


avaliadorDemodelo(
    rf_final,
    atributos_teste,
    target_teste,
    "Random Forest"
)

avaliadorDemodelo(
    svc_final,
    atributos_teste,
    target_teste,
    "SVC"
)
"""
Fitting 5 folds for each of 20 candidates, totalling 100 fits
Melhores par�metros rf: {'n_estimators': 500, 'min_samples_split': 10, 'max_features': 'log2', 'max_depth': 47, 'criterion': 'gini'}
Melhor score CV rf: 0.9140360691325039
Melhores par�metros svc: {'kernel': 'poly', 'gamma': 'auto', 'degree': 5, 'coef0': 0.5, 'C': 10}
Melhor score CV svc: 0.8910160263311699
Tempo de treino RF: 0.2661 segundos
Tempo de treino SVC: 0.0054 segundos

//////////////////////////////////////////////////////////////////////////////////////////////////////
Random Forest

//////////////////////////////////////////////////////////////////////////////////////////////////////
Accrcy: 0.8111111111111111
F(uma)1: 0.7017543859649122
ROCAUC: 0.9005087620124365

Matriz de Confus�o
[[53  8]
 [ 9 20]]

Relat�rio
              precision    recall  f1-score   support

           0       0.85      0.87      0.86        61
           1       0.71      0.69      0.70        29

    accuracy                           0.81        90
   macro avg       0.78      0.78      0.78        90
weighted avg       0.81      0.81      0.81        90


//////////////////////////////////////////////////////////////////////////////////////////////////////
SVC

//////////////////////////////////////////////////////////////////////////////////////////////////////
Accrcy: 0.7777777777777778
F(uma)1: 0.6
ROCAUC: 0.8134539287733181

Matriz de Confus�o
[[55  6]
 [14 15]]

Relat�rio
              precision    recall  f1-score   support

           0       0.80      0.90      0.85        61
           1       0.71      0.52      0.60        29

    accuracy                           0.78        90
   macro avg       0.76      0.71      0.72        90
weighted avg       0.77      0.78      0.77        90'''

#ta eu ja suspeitava que random forest ia ser melhor mas achei bom testar, como tava na ementa eu suspeitei que as variaveis binarias iam dar uma dificultada mas como era
#tudo numero achei bom testar a vector, mas mesmo assim as arvores conseguem lidar mt bem com dados binarios enquantoa vm tem aquele negocio de superfice geometrica
# onde esse tipo de dado acaba sendo meio bunda quando apresentado dessa forma pq ela n consegue explorar tao bem as divisoes na fronteira de decisao
#ai tmbm ja entra na minha especulacao com fonte vozes da minha cabeça, mas eu acho que de forma geral as features do dataset sao mt mais bem exploraveis por random forest do q por
#vector machine pelo simples fato que o comportamento delas na hora de predição acaba sendo melhor pras trees do que pro modelo matematico / cartesiano da vector machine 
"""
pickle.dump(rf_final, open("JARRO DE PICLES(pickles)/HeartAtackRF.pkl", "wb"))
