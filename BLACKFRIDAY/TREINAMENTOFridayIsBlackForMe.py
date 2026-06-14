import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from ModuloAvaliador import avaliadorDemodelo
import pickle
import time

#eu fiqueio um bom tempo batendo a cabeça de como eu ia fazer pra processar um csv desse tamanho, ent determinei que a melhor ideia seria quebrar ele
# em varias partes, antes que o meu pc pegasse fogo, e ainda demorou 10 minutos USANDO XGB E RANDOM
# fazer tudo de uma vez com random forest e GRID teria sido meui fim

# funcao de timer pq eu queria pagar pra ver o tmepo q isso iria demorar
def timer(label):
    start = time.time()
    print(f"\nstart {label}...")
    def stop():
        elapsed = time.time() - start
        print(f"cabo {label} — {elapsed:.2f}s ({elapsed/60:.1f} min)")
    return stop

dados = pd.read_csv("DATASETS/retail_black_friday_sales_100k.csv")
dados_original = dados.copy()
dados = dados.drop(columns=["transaction_id", "customer_id", "product_id", "purchase_date"])

ordem_age = {"18-25": 0, "26-35": 1, "36-45": 2, "46-55": 3, "56+": 4}
dados["age_group_enc"] = dados["age_group"].map(ordem_age)

dados_dummies = pd.get_dummies(dados, columns=["gender", "city", "customer_segment"], drop_first=False)

print(f"Dataset: {dados.shape[0]} amostras")

le_cat = LabelEncoder()
le_pag = LabelEncoder()
le_age = LabelEncoder()

target_categoria = le_cat.fit_transform(dados_original["product_category"])
target_pagamento = le_pag.fit_transform(dados_original["payment_method"])
target_idade     = le_age.fit_transform(dados_original["age_group"])

print(f"product_category: {list(le_cat.classes_)}")
print(f"payment_method:   {list(le_pag.classes_)}")
print(f"age_group:        {list(le_age.classes_)}\n")


features_cat = dados_dummies.drop(columns=["product_category", "payment_method", "age_group"])
features_pag = dados_dummies.drop(columns=["product_category", "payment_method", "age_group"])
features_age = dados_dummies.drop(columns=["product_category", "payment_method", "age_group", "age_group_enc"])


grade_xgb = {
    "n_estimators": [int(x) for x in np.linspace(100, 800, 8)],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 1.5, 2]
}

#sim eu usei o mesmo grid pra todo mundo pq considerei q n faria mt diferença pra eles depois de eu tunar ele

class WrapperBinario:
    def __init__(self, modelo, ref):
        self.modelo = modelo
        self.ref = ref
    def predict(self, X):
        return (self.modelo.predict(X) == self.ref).astype(int)
    def predict_proba(self, X):
        p = self.modelo.predict_proba(X)[:, self.ref]
        return np.column_stack([1 - p, p])

# wrapper pro xgb nn explodir em 340 mil ep0daços

t_total = timer("Pipeline — product_category")

X_train, X_test, y_train, y_test = train_test_split(
    features_cat, target_categoria, test_size=0.3, random_state=42, stratify=target_categoria
)

scaler_cat = StandardScaler()
X_train = scaler_cat.fit_transform(X_train)
X_test  = scaler_cat.transform(X_test)

menor_classe = pd.Series(y_train).value_counts().min()
k_viz = min(5, menor_classe - 1)
smoter = SMOTE(random_state=42, k_neighbors=k_viz)
X_train_b, y_train_b = smoter.fit_resample(X_train, y_train)
print(f"Apos SMOTE: {X_train_b.shape[0]} amostras")

t_cv = timer("RandomizedSearchCV — product_category")
random_search_cat = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss"),
    param_distributions=grade_xgb,
    n_iter=30, cv=5, scoring="f1_weighted",
    n_jobs=-1, random_state=42, verbose=1
)
random_search_cat.fit(X_train_b, y_train_b)
print(f"Melhores params product_category: {random_search_cat.best_params_}")
print(f"Melhor CV f1_weighted: {random_search_cat.best_score_:.4f}")
t_cv()

t_fit = timer("Treino final — product_category")
modelo_cat = XGBClassifier(**random_search_cat.best_params_, random_state=42, n_jobs=-1, eval_metric="mlogloss")
modelo_cat.fit(X_train_b, y_train_b)
t_fit()

classe_ref_cat = int(pd.Series(y_test).value_counts().idxmax())
y_test_bin_cat = (y_test == classe_ref_cat).astype(int)
classe_nome_cat = le_cat.inverse_transform([classe_ref_cat])[0]

res_cat = avaliadorDemodelo(WrapperBinario(modelo_cat, classe_ref_cat), X_test, y_test_bin_cat, f"product_category (ref: {classe_nome_cat})")

cm = confusion_matrix(y_test_bin_cat, WrapperBinario(modelo_cat, classe_ref_cat).predict(X_test))
tn, fp, fn, tp = cm.ravel()
print(f"Sensibilidade (Recall): {tp / (tp + fn):.4f}")
print(f"Especificidade:         {tn / (tn + fp):.4f}")

t_total()

t_total = timer("Pipeline — payment_method")

X_train, X_test, y_train, y_test = train_test_split(
    features_pag, target_pagamento, test_size=0.3, random_state=42, stratify=target_pagamento
)

scaler_pag = StandardScaler()
X_train = scaler_pag.fit_transform(X_train)
X_test  = scaler_pag.transform(X_test)

menor_classe = pd.Series(y_train).value_counts().min()
k_viz = min(5, menor_classe - 1)
smoter = SMOTE(random_state=42, k_neighbors=k_viz)
X_train_b, y_train_b = smoter.fit_resample(X_train, y_train)
print(f"Apos SMOTE: {X_train_b.shape[0]} amostras")

t_cv = timer("RandomizedSearchCV — payment_method")
random_search_pag = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss"),
    param_distributions=grade_xgb,
    n_iter=30, cv=5, scoring="f1_weighted",
    n_jobs=-1, random_state=42, verbose=1
)
random_search_pag.fit(X_train_b, y_train_b)
print(f"Melhores params payment_method: {random_search_pag.best_params_}")
print(f"Melhor CV f1_weighted: {random_search_pag.best_score_:.4f}")
t_cv()

t_fit = timer("Treino final — payment_method")
modelo_pag = XGBClassifier(**random_search_pag.best_params_, random_state=42, n_jobs=-1, eval_metric="mlogloss")
modelo_pag.fit(X_train_b, y_train_b)
t_fit()

classe_ref_pag = int(pd.Series(y_test).value_counts().idxmax())
y_test_bin_pag = (y_test == classe_ref_pag).astype(int)
classe_nome_pag = le_pag.inverse_transform([classe_ref_pag])[0]

res_pag = avaliadorDemodelo(WrapperBinario(modelo_pag, classe_ref_pag), X_test, y_test_bin_pag, f"payment_method (ref: {classe_nome_pag})")

cm = confusion_matrix(y_test_bin_pag, WrapperBinario(modelo_pag, classe_ref_pag).predict(X_test))
tn, fp, fn, tp = cm.ravel()
print(f"Sensibilidade: {tp / (tp + fn):.4f}")
print(f"Especificidade:{tn / (tn + fp):.4f}")

t_total()

t_total = timer("Pipeline — age_group")

X_train, X_test, y_train, y_test = train_test_split(
    features_age, target_idade, test_size=0.3, random_state=42, stratify=target_idade
)

scaler_age = StandardScaler()
X_train = scaler_age.fit_transform(X_train)
X_test  = scaler_age.transform(X_test)

menor_classe = pd.Series(y_train).value_counts().min()
k_viz = min(5, menor_classe - 1)
smoter = SMOTE(random_state=42, k_neighbors=k_viz)
X_train_b, y_train_b = smoter.fit_resample(X_train, y_train)
print(f"Apos SMOTE: {X_train_b.shape[0]} amostras")

t_cv = timer("RandomizedSearchCV — age_group")
random_search_age = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss"),
    param_distributions=grade_xgb,
    n_iter=30, cv=5, scoring="f1_weighted",
    n_jobs=-1, random_state=42, verbose=1
)
random_search_age.fit(X_train_b, y_train_b)
print(f"Melhores params age_group: {random_search_age.best_params_}")
print(f"Melhor CV f1_weighted: {random_search_age.best_score_:.4f}")
t_cv()

t_fit = timer("Treino final — age_group")
modelo_age = XGBClassifier(**random_search_age.best_params_, random_state=42, n_jobs=-1, eval_metric="mlogloss")
modelo_age.fit(X_train_b, y_train_b)
t_fit()

classe_ref_age = int(pd.Series(y_test).value_counts().idxmax())
y_test_bin_age = (y_test == classe_ref_age).astype(int)
classe_nome_age = le_age.inverse_transform([classe_ref_age])[0]

res_age = avaliadorDemodelo(WrapperBinario(modelo_age, classe_ref_age), X_test, y_test_bin_age, f"age_group (ref: {classe_nome_age})")

cm = confusion_matrix(y_test_bin_age, WrapperBinario(modelo_age, classe_ref_age).predict(X_test))
tn, fp, fn, tp = cm.ravel()
print(f"Sensibilidade (Recall): {tp / (tp + fn):.4f}")
print(f"Especificidade:         {tn / (tn + fp):.4f}")

t_total()


df_resultados = pd.DataFrame([res_cat, res_pag, res_age]).set_index("Modelo")
print("\n\nCOMPARATIVO FINAL")
print(df_resultados.to_string())


pickle.dump({"modelo": modelo_cat, "scaler": scaler_cat, "le": le_cat, "features": list(features_cat.columns)}, open("JARRO DE PICLES(pickles)/BF_categoria.pkl", "wb"))
pickle.dump({"modelo": modelo_pag, "scaler": scaler_pag, "le": le_pag, "features": list(features_pag.columns)}, open("JARRO DE PICLES(pickles)/BF_pagamento.pkl", "wb"))
pickle.dump({"modelo": modelo_age, "scaler": scaler_age, "le": le_age, "features": list(features_age.columns)}, open("JARRO DE PICLES(pickles)/BF_idade.pkl", "wb"))
print("salvo salvado")


"""ataset: 100000 amostras
product_category: ['Accessories', 'Beauty', 'Books', 'Clothing', 'Electronics', 'Footwear', 'Groceries', 'Home & Kitchen', 'Sports', 'Toys']
payment_method:   ['Cash', 'Credit Card', 'Debit Card', 'Gift Card', 'Mobile Wallet', 'PayPal']
age_group:        ['18-25', '26-35', '36-45', '46-55', '56+']


[INICIO] Pipeline � product_category...
Apos SMOTE: 71090 amostras

[INICIO] RandomizedSearchCV � product_category...
Fitting 5 folds for each of 30 candidates, totalling 150 fits
Melhores params product_category: {'subsample': 0.9, 'reg_lambda': 2, 'reg_alpha': 0.1, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 4, 'learning_rate': 0.1, 'gamma': 0.5, 'colsample_bytree': 0.8}
Melhor CV f1_weighted: 0.3280
[FIM]    RandomizedSearchCV � product_category � 323.17s (5.4 min)

[INICIO] Treino final � product_category...
[FIM]    Treino final � product_category � 1.69s (0.0 min)

//////////////////////////////////////////////////////////////////////////////////////////////////////
product_category (ref: Accessories)

//////////////////////////////////////////////////////////////////////////////////////////////////////
Accrcy: 0.8848666666666667
F(uma)1: 0.2948142098815843
ROCAUC: 0.7608919772839691

Matriz de Confus�o
[[25824  1129]
 [ 2325   722]]

Relat�rio
              precision    recall  f1-score   support

           0       0.92      0.96      0.94     26953
           1       0.39      0.24      0.29      3047

    accuracy                           0.88     30000
   macro avg       0.65      0.60      0.62     30000
weighted avg       0.86      0.88      0.87     30000

Sensibilidade (Recall): 0.2370
Especificidade:         0.9581
[FIM]    Pipeline � product_category � 326.75s (5.4 min)

[INICIO] Pipeline � payment_method...
Apos SMOTE: 70440 amostras

[INICIO] RandomizedSearchCV � payment_method...
Fitting 5 folds for each of 30 candidates, totalling 150 fits
Melhores params payment_method: {'subsample': 0.7, 'reg_lambda': 1, 'reg_alpha': 0.1, 'n_estimators': 600, 'min_child_weight': 5, 'max_depth': 4, 'learning_rate': 0.05, 'gamma': 0.2, 'colsample_bytree': 0.9}
Melhor CV f1_weighted: 0.1660
[FIM]    RandomizedSearchCV � payment_method � 189.38s (3.2 min)

[INICIO] Treino final � payment_method...
[FIM]    Treino final � payment_method � 3.49s (0.1 min)

//////////////////////////////////////////////////////////////////////////////////////////////////////
payment_method (ref: Debit Card)

//////////////////////////////////////////////////////////////////////////////////////////////////////
Accrcy: 0.7303
F(uma)1: 0.16510164069755442
ROCAUC: 0.5051021428254876

Matriz de Confus�o
[[21109  3859]
 [ 4232   800]]

Relat�rio
              precision    recall  f1-score   support

           0       0.83      0.85      0.84     24968
           1       0.17      0.16      0.17      5032

    accuracy                           0.73     30000
   macro avg       0.50      0.50      0.50     30000
weighted avg       0.72      0.73      0.73     30000

Sensibilidade (Recall): 0.1590
Especificidade:         0.8454
[FIM]    Pipeline � payment_method � 193.59s (3.2 min)

[INICIO] Pipeline � age_group...
Apos SMOTE: 70505 amostras

[INICIO] RandomizedSearchCV � age_group...
Fitting 5 folds for each of 30 candidates, totalling 150 fits
Melhores params age_group: {'subsample': 1.0, 'reg_lambda': 2, 'reg_alpha': 0.1, 'n_estimators': 600, 'min_child_weight': 3, 'max_depth': 5, 'learning_rate': 0.05, 'gamma': 0.1, 'colsample_bytree': 0.9}
Melhor CV f1_weighted: 0.2016
[FIM]    RandomizedSearchCV � age_group � 147.05s (2.5 min)

[INICIO] Treino final � age_group...
[FIM]    Treino final � age_group � 1.91s (0.0 min)

//////////////////////////////////////////////////////////////////////////////////////////////////////
age_group (ref: 36-45)

//////////////////////////////////////////////////////////////////////////////////////////////////////
Accrcy: 0.6947333333333333
F(uma)1: 0.194971870604782
ROCAUC: 0.5100415833782628

Matriz de Confus�o
[[19733  4223]
 [ 4935  1109]]

Relat�rio
              precision    recall  f1-score   support

           0       0.80      0.82      0.81     23956
           1       0.21      0.18      0.19      6044

    accuracy                           0.69     30000
   macro avg       0.50      0.50      0.50     30000
weighted avg       0.68      0.69      0.69     30000

Sensibilidade (Recall): 0.1835
Especificidade:         0.8237
[FIM]    Pipeline � age_group � 149.63s (2.5 min)


COMPARATIVO FINAL
                                     Accuracy        F1   ROC_AUC
Modelo                                                           
product_category (ref: Accessories)  0.884867  0.294814  0.760892
payment_method (ref: Debit Card)     0.730300  0.165102  0.505102
age_group (ref: 36-45)               0.694733  0.194972  0.510042
salvo salvado"""