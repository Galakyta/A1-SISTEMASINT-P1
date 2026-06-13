import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from ModuloAvaliador import avaliadorDemodelo
import pickle
import time
 

 #fiquei com medo do tyamanho do csv ent decidi melhorar o timer dele tmbm, v usar xgboost pq nem FERRANDO que eu vo usar random forest nisso
 # mesmo que ela dispense os tratamentos ia demorar MUUUUUUUITO, ent preferi xgboost 
def timer(label):
    start = time.time()
    print(f"\n[INICIO] {label}...")
    def stop():
        elapsed = time.time() - start
        print(f"[FIM]    {label} — {elapsed:.2f}s ({elapsed/60:.1f} min)")
    return stop
 

#ok esse dataset tava bem dificinho mas acho q me virei, mesmo COM 100K LINHAS
dados = pd.read_csv("DATASETS/retail_black_friday_sales_100k.csv")
dados_original = dados.copy()
dados = dados.drop(columns=["transaction_id", "customer_id", "product_id", "purchase_date"])
 
ordem_age = {"18-25": 0, "26-35": 1, "36-45": 2, "46-55": 3, "56+": 4}
dados["age_group_enc"] = dados["age_group"].map(ordem_age)
 
dados_dummies = pd.get_dummies(dados, columns=["gender", "city", "customer_segment"], drop_first=False)
 
print(f"Dataset: {dados.shape[0]} amostras")
print(dados["payment_method"].value_counts())
print(dados["age_group"].value_counts())