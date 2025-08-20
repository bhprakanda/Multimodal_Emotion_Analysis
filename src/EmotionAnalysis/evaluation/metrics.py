from sklearn.metrics import classification_report, f1_score, accuracy_score

def generate_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=4)
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    return report, f1, acc