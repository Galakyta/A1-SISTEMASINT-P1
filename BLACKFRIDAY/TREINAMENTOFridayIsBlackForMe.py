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

def timer(label):
    start = time.time()
    print(f"\n[INICIO] {label}...")
    def stop():
        elapsed = time.time() - start
        print(f"[FIM]    {label} — {elapsed:.2f}s ({elapsed/60:.1f} min)")
    return stop

# ---------------------------------------------------------------
# 1. CARREGAR E PREPARAR
# ---------------------------------------------------------------
dados = pd.read_csv("DATASETS/retail_black_friday_sales_100k.csv")
dados_original = dados.copy()
dados = dados.drop(columns=["transaction_id", "customer_id", "product_id", "purchase_date"])

ordem_age = {"18-25": 0, "26-35": 1, "36-45": 2, "46-55": 3, "56+": 4}
dados["age_group_enc"] = dados["age_group"].map(ordem_age)

dados_dummies = pd.get_dummies(dados, columns=["gender", "city", "customer_segment"], drop_first=False)

print(f"Dataset: {dados.shape[0]} amostras")

# ---------------------------------------------------------------
# 2. TARGETS
# ---------------------------------------------------------------
le_cat = LabelEncoder()
le_pag = LabelEncoder()
le_age = LabelEncoder()

target_categoria = le_cat.fit_transform(dados_original["product_category"])
target_pagamento = le_pag.fit_transform(dados_original["payment_method"])
target_idade     = le_age.fit_transform(dados_original["age_group"])

print(f"product_category: {list(le_cat.classes_)}")
print(f"payment_method:   {list(le_pag.classes_)}")
print(f"age_group:        {list(le_age.classes_)}\n")

# ---------------------------------------------------------------
# 3. FEATURES POR TARGET
# ---------------------------------------------------------------
features_cat = dados_dummies.drop(columns=["product_category", "payment_method", "age_group"])
features_pag = dados_dummies.drop(columns=["product_category", "payment_method", "age_group"])
features_age = dados_dummies.drop(columns=["product_category", "payment_method", "age_group", "age_group_enc"])

# ---------------------------------------------------------------
# 4. GRADE DE PARAMETROS
# ---------------------------------------------------------------
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

class WrapperBinario:
    def __init__(self, modelo, ref):
        self.modelo = modelo
        self.ref = ref
    def predict(self, X):
        return (self.modelo.predict(X) == self.ref).astype(int)
    def predict_proba(self, X):
        p = self.modelo.predict_proba(X)[:, self.ref]
        return np.column_stack([1 - p, p])

# ---------------------------------------------------------------
# 5. PRODUCT CATEGORY
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 6. PAYMENT METHOD
# ---------------------------------------------------------------
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
print(f"Sensibilidade (Recall): {tp / (tp + fn):.4f}")
print(f"Especificidade:         {tn / (tn + fp):.4f}")

t_total()

# ---------------------------------------------------------------
# 7. AGE GROUP
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 8. COMPARATIVO
# ---------------------------------------------------------------
df_resultados = pd.DataFrame([res_cat, res_pag, res_age]).set_index("Modelo")
print("\n\nCOMPARATIVO FINAL")
print(df_resultados.to_string())

# ---------------------------------------------------------------
# 9. SALVAR
# ---------------------------------------------------------------
pickle.dump({"modelo": modelo_cat, "scaler": scaler_cat, "le": le_cat, "features": list(features_cat.columns)}, open("JARRO DE PICLES(pickles)/BF_categoria.pkl", "wb"))
pickle.dump({"modelo": modelo_pag, "scaler": scaler_pag, "le": le_pag, "features": list(features_pag.columns)}, open("JARRO DE PICLES(pickles)/BF_pagamento.pkl", "wb"))
pickle.dump({"modelo": modelo_age, "scaler": scaler_age, "le": le_age, "features": list(features_age.columns)}, open("JARRO DE PICLES(pickles)/BF_idade.pkl", "wb"))
print("salvo salvado")