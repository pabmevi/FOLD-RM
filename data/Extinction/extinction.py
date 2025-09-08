import sys
import random
random.seed(42)
sys.path.insert(0, "/home/pabmevi/CONFOLD")

import numpy as np
from foldrm import Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

def extinction():
    attrs = ["IslandEndemic","Volancy","Mass","HWI","Habitat","Trophic.Level",
          "Trophic.Niche","LAT","Beak.Length.culmen",
          "Beak.Length.nares","Beak.Width","Beak.Depth","Tarsus.Length",
          "Wing.Length","Kipps.Distance","Secondary1","Tail.Length"]
    nums = ["Mass","HWI","LAT","Beak.Length.culmen",
          "Beak.Length.nares","Beak.Width","Beak.Depth","Tarsus.Length",
          "Wing.Length","Kipps.Distance","Secondary1","Tail.Length"]

    model = Classifier(attrs=attrs, numeric=nums, label='status_group')
    data = model.load_data('/home/pabmevi/CONFOLD/FOLD-RM/data/Extinction/BirdstraitsIUCN.csv')
    print('\n% traits dataset', np.shape(data))
    # Filtrar solo clases válidas
    valid_labels = {'Not threatened', 'Threatened', 'DD'}
    filtered_data = [row for row in data if str(row[-1]).strip() in valid_labels]
    print(f"\nFiltrados {len(data) - len(filtered_data)} registros con clases no válidas.")
    return model, filtered_data

model, data = extinction()

from utils import split_data
train_data, test_data = split_data(data, ratio=0.9, shuffle=True)

# ===========================
# Mostrar distribución de clases en train y test
# ===========================
def print_class_distribution(dataset, name):
    labels = [row[-1] for row in dataset]
    dist = pd.Series(labels).value_counts()
    print(f"\nDistribución de clases en {name}:")
    print(dist)

print_class_distribution(train_data, "train")
print_class_distribution(test_data, "test")

# ===========================
# Training (parámetros menos estrictos para mejorar cobertura)
# ===========================
model.fit(train_data, ratio=0.9)
model.confidence_fit(train_data, improvement_threshold=0.9)

print("\nLearned Answer Set Program rules:\n")
model.print_asp()

# ===========================
# Predicting over test_data
# ===========================
Y_pred = model.predict(test_data)

print("\nEjemplo de predicciones (primeros 10):")
for i, (pred, obs) in enumerate(zip(Y_pred[:10], test_data[:10])):
    print(f"Obs {i+1}: pred = {pred}, entrada = {obs}")

# ===========================
# Evaluación del modelo
# ===========================
# Accuracy global (cuenta None como error)
all_pred_classes = [p[0] if p is not None else None for p in Y_pred]
all_true_classes = [row[-1] for row in test_data]
acc_global = sum([y1 == y2 for y1, y2 in zip(all_pred_classes, all_true_classes)]) / len(all_true_classes)
print("\nAccuracy global (incluyendo None como error):", acc_global)

# Extraer clases predichas y etiquetas reales, filtrando None
pred_classes = [p[0] for p in Y_pred if p is not None and p[0] is not None]
true_classes = [row[-1] for p, row in zip(Y_pred, test_data) if p is not None and p[0] is not None]

# Accuracy general
if pred_classes:
    acc = accuracy_score(true_classes, pred_classes)
    print("\nAccuracy general:", acc)
else:
    print("\nNo hay predicciones válidas para calcular accuracy.")

# Matriz de confusión
labels = ['threatened', 'Not threatened', 'dd']
if pred_classes:
    cm = confusion_matrix(true_classes, pred_classes, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nMatriz de confusión:")
    print(df_cm)
else:
    print("\nNo hay predicciones válidas para matriz de confusión.")



