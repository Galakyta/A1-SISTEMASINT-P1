from pickle import load
import pandas as pd
import numpy as np

modelo_xgb = load(open("JARRO DE PICLES(pickles)/VinhoXGB.pkl", "rb"))
scaler = load(open("JARRO DE PICLES(pickles)/VinhoScaler.pkl", "rb"))
le = load(open("JARRO DE PICLES(pickles)/VinhoLabelEncoder.pkl", "rb"))

# aq da pra substituir por qualquer vinho
# eu forcei ao maximo pra dar um vinho mt bom mas o resultado foi meh
novo_vinho = pd.DataFrame([{
    "fixed acidity": 6.0,
    "volatile acidity": 0.10,   
    "citric acid": 0.45,
    "residual sugar": 1.2,
    "chlorides": 0.020,         
    "free sulfur dioxide": 25.0,
    "total sulfur dioxide": 60.0, 
    "density": 0.9880,          
    "pH": 3.1,
    "sulphates": 0.90,
    "alcohol": 14.5,            
    "wine_type": 1
}])

novo_vinho_scaled = scaler.transform(novo_vinho)

pred_enc = modelo_xgb.predict(novo_vinho_scaled)[0]
proba_todas = modelo_xgb.predict_proba(novo_vinho_scaled)[0]

pred_quality = le.inverse_transform([pred_enc])[0]

threshold_enc = int(le.transform([7])[0])
prob_bom = proba_todas[threshold_enc:].sum()
prob_nao_bom = 1 - prob_bom

print("Resultado XGBoost — Wine Quality:")
print(f"Predicao: quality {pred_quality} ({'bom' if pred_quality >= 7 else 'nao bom'})")
print(f"Probabilidade bom (quality >= 7): {prob_bom:.2%}")
print(f"Probabilidade nao bom (quality < 7): {prob_nao_bom:.2%}")

"""Resultado XGBoost � Wine Quality:
Predicao: quality 6 (nao bom)
Probabilidade bom (quality >= 7): 31.90%
Probabilidade nao bom (quality < 7): 68.10%
"""
#eu acho q o modelo ficou bem conservador pela vasta maioria dos vinhos no csv serem bem mais ou menos e ficarem ali entre 5 e 7, oq faz sentido ja q tem mt pouco
#com 9, acho q nove foram só 3 por isso q deu erro no meu smote inclusive, ent o modelo é conservador mas faz sentido ser 