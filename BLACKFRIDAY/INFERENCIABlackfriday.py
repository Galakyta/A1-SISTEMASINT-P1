from pickle import load
import pandas as pd
import numpy as np
import os
print(os.getcwd())
bf_cat = load(open("JARRO DE PICLES(pickles)/BF_categoria.pkl", "rb"))
bf_pag = load(open("JARRO DE PICLES(pickles)/BF_pagamento.pkl", "rb"))
bf_age = load(open("JARRO DE PICLES(pickles)/BF_idade.pkl",     "rb"))


nova_venda = {
    "age_group": "26-35",
    "gender": "Male",
    "city": "San Francisco",
    "customer_segment": "Loyal",
    "original_price": 399.99,
    "discount_pct": 30,
    "final_price": 279.99,
    "quantity": 2,
    "purchase_amount": 559.98,
    "purchase_hour": 14,
    "is_weekend": 1,
    "is_black_friday": 1
}

ordem_age = {"18-25": 0, "26-35": 1, "36-45": 2, "46-55": 3, "56+": 4}

def preparar(venda, features_esperadas, incluir_age_enc=True):
    df = pd.DataFrame([venda])
    if incluir_age_enc:
        df["age_group_enc"] = df["age_group"].map(ordem_age)
    df = df.drop(columns=["age_group"], errors="ignore")
    df = pd.get_dummies(df, columns=["gender", "city", "customer_segment"], drop_first=False)
    df = df.reindex(columns=features_esperadas, fill_value=0)
    return df

# ---------------------------------------------------------------
# product_category
# ---------------------------------------------------------------
X_cat = preparar(nova_venda, bf_cat["features"], incluir_age_enc=True)
X_cat_scaled = bf_cat["scaler"].transform(X_cat)

pred_cat_enc = bf_cat["modelo"].predict(X_cat_scaled)[0]
proba_cat    = bf_cat["modelo"].predict_proba(X_cat_scaled)[0]
pred_cat     = bf_cat["le"].inverse_transform([pred_cat_enc])[0]

print("Resultado — Product Category:")
print(f"Predicao: {pred_cat}")
print(f"Grau de certeza: {proba_cat[pred_cat_enc]:.2%}")
print("Probabilidades por classe:")
for i, classe in enumerate(bf_cat["le"].classes_):
    barra = "#" * int(proba_cat[i] * 40)
    print(f"  {classe:<18} {proba_cat[i]:.2%}  {barra}")

# ---------------------------------------------------------------
# payment_method
# ---------------------------------------------------------------
X_pag = preparar(nova_venda, bf_pag["features"], incluir_age_enc=True)
X_pag_scaled = bf_pag["scaler"].transform(X_pag)

pred_pag_enc = bf_pag["modelo"].predict(X_pag_scaled)[0]
proba_pag    = bf_pag["modelo"].predict_proba(X_pag_scaled)[0]
pred_pag     = bf_pag["le"].inverse_transform([pred_pag_enc])[0]

print("\nResultado — Payment Method:")
print(f"Predicao: {pred_pag}")
print(f"Grau de certeza: {proba_pag[pred_pag_enc]:.2%}")
print("Probabilidades por classe:")
for i, classe in enumerate(bf_pag["le"].classes_):
    barra = "#" * int(proba_pag[i] * 40)
    print(f"  {classe:<18} {proba_pag[i]:.2%}  {barra}")

# ---------------------------------------------------------------
# age_group
# ---------------------------------------------------------------
X_age = preparar(nova_venda, bf_age["features"], incluir_age_enc=False)
X_age_scaled = bf_age["scaler"].transform(X_age)

pred_age_enc = bf_age["modelo"].predict(X_age_scaled)[0]
proba_age    = bf_age["modelo"].predict_proba(X_age_scaled)[0]
pred_age     = bf_age["le"].inverse_transform([pred_age_enc])[0]

print("\nResultado — Age Group:")
print(f"Predicao: {pred_age}")
print(f"Grau de certeza: {proba_age[pred_age_enc]:.2%}")
print("Probabilidades por classe:")
for i, classe in enumerate(bf_age["le"].classes_):
    barra = "#" * int(proba_age[i] * 40)
    print(f"  {classe:<18} {proba_age[i]:.2%}  {barra}")