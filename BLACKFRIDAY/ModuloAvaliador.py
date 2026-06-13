from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def avaliadorDemodelo(modelo, X_teste, y_teste, nome="Modelo"):

    pred = modelo.predict(X_teste)

    print("\n//////////////////////////////////////////////////////////////////////////////////////////////////////")
    print(nome)
    print("\n//////////////////////////////////////////////////////////////////////////////////////////////////////")

    print("Accrcy:",
          accuracy_score(y_teste, pred))

    print("F(uma)1:",
          f1_score(y_teste, pred))

    if hasattr(modelo, "predict_proba"):
        prob = modelo.predict_proba(X_teste)[:, 1]

        print("ROCAUC:",
              roc_auc_score(y_teste, prob))

    print("\nMatriz de Confusão")
    print(confusion_matrix(y_teste, pred))

    print("\nRelatório")
    print(classification_report(y_teste, pred))
    pred = modelo.predict(X_teste)
    prob = modelo.predict_proba(X_teste)[:, 1]
    resultado = {
        "Modelo": nome,
        "Accuracy": accuracy_score(y_teste, pred),
        "F1": f1_score(y_teste, pred),
        "ROC_AUC": roc_auc_score(y_teste, prob)
    }
    return resultado