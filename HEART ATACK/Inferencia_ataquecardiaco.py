from pickle import load
import pandas as pd
import numpy as np

modelo_rf = load(open("JARRO DE PICLES(pickles)/HeartAtackRF.pkl", "rb"))


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dados = pd.read_csv("DATASETS/heart_failure_clinical_records_dataset.csv", sep=",")
dados_atributos = dados.drop(columns=["DEATH_EVENT"])
target = dados["DEATH_EVENT"]

_, atributos_teste_orig, _, _ = train_test_split(
    dados_atributos, target, test_size=0.3, random_state=42, stratify=target
)

scaler = StandardScaler()
scaler.fit_transform(dados_atributos)  # fita no total igual no treino original

# aq da pra substituir por qualquer paciente, mas eu forcei um que o cara VAI MORRE
novo_paciente = pd.DataFrame([{
    "age": 60,
    "anaemia": 1,
    "creatinine_phosphokinase": 582,
    "diabetes": 0,
    "ejection_fraction": 20,        
    "high_blood_pressure": 1,
    "platelets": 265000,
    "serum_creatinine": 1.9,        
    "serum_sodium": 130,            
    "sex": 1,
    "smoking": 0,
    "time": 4                       
}])

novo_paciente_scaled = scaler.transform(novo_paciente)


pred_rf = modelo_rf.predict(novo_paciente_scaled)[0]
proba_rf = modelo_rf.predict_proba(novo_paciente_scaled)[0]

print("Resultado Random Forest:")
print(f"Predicao: {'obito' if pred_rf == 1 else 'sobreviveu'}")
print(f"Probabilidade sobreviveu: {proba_rf[0]:.2%}")
print(f"Probabilidade obito:      {proba_rf[1]:.2%}")