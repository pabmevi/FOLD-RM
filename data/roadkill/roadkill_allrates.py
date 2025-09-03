# ===========================
# Evaluación del modelo (robusta)
# ===========================

total_preds = len(Y_pred)
none_preds = sum(1 for pred, _ in Y_pred if pred is None)

# Filtrar predicciones válidas
filtered_true = []
filtered_pred = []
filtered_conf = []

for t, (pred, conf) in zip(true_classes, Y_pred):
    if pred is not None:
        filtered_true.append(t)
        filtered_pred.append(pred)
        filtered_conf.append(conf)

print(f"\nTotal de observaciones: {total_preds}")
print(f"Predicciones válidas: {len(filtered_true)}")
print(f"Predicciones descartadas por None: {none_preds}")
print(f"Predicciones descartadas por confianza: {total_preds - len(filtered_true)}\n")

# Accuracy general
acc = accuracy_score(filtered_true, filtered_pred)
print("Accuracy general:", acc)

# Matriz de confusión
labels = ['low', 'medium', 'high']
cm = confusion_matrix(filtered_true, filtered_pred, labels=labels)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
print("\nMatriz de confusión:")
print(df_cm)

# Reporte de precisión, recall y f1-score
print("\nReporte de clasificación:")
print(classification_report(filtered_true, filtered_pred, labels=labels))

# Accuracy de predicciones de alta confianza (>= 0.8)
high_conf_preds = [(pred, true) for pred, true, conf in zip(filtered_pred, filtered_true, filtered_conf) if conf >= 0.8]
num_high_conf = len(high_conf_preds)

if num_high_conf > 0:
    accuracy_high_conf = sum(1 for (pred, true) in high_conf_preds if pred == true) / num_high_conf
    print(f"\nPredicciones con confianza >= 0.8: {num_high_conf}")
    print("Accuracy para predicciones de alta confianza:", accuracy_high_conf)
else:
    print("\nNo hay predicciones con confianza >= 0.8")
