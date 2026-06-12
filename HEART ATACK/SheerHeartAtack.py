import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


