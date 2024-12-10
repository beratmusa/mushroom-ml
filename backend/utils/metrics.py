from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)


def calculate_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        "confusion_matrix": conf_matrix.tolist(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
