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
    # Identificar registros con clases no válidas
    valid_labels = {'Not threatened', 'Threatened', 'DD'}
    invalid = [(i, row[-1]) for i, row in enumerate(data) if str(row[-1]).strip() not in valid_labels]
    if invalid:
        print(f"\nRegistros con status_group no válido ({len(invalid)} casos):")
        for idx, val in invalid:
            print(f"Índice {idx}: status_group = {val}")
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

# ===========================
# Balancear el conjunto de entrenamiento
# ===========================
def balance_data(data):
    threatened = [row for row in data if row[-1] == 'Threatened']
    not_threatened = [row for row in data if row[-1] == 'Not threatened']
    dd = [row for row in data if row[-1] == 'DD']
    n = min(len(threatened), len(not_threatened))
    # Submuestreo aleatorio de Not threatened y Threatened
    random.seed(42)
    not_threatened_sample = random.sample(not_threatened, n)
    threatened_sample = random.sample(threatened, n)
    balanced = threatened_sample + not_threatened_sample + dd
    random.shuffle(balanced)
    return balanced


print_class_distribution(train_data, "train (original)")
print_class_distribution(test_data, "test")


# Balancear y mostrar nueva distribución
balanced_train_data = balance_data(train_data)
balanced_test_data = balance_data(test_data)

print_class_distribution(balanced_train_data, "train (balanceado)")
print_class_distribution(balanced_test_data, "test (balanceado)")

# Verificar clases en test balanceado antes de predecir
labels_test = [row[-1] for row in balanced_test_data]
print(f"\nConteo Threatened en test balanceado: {labels_test.count('Threatened')}")
print(f"Conteo Not threatened en test balanceado: {labels_test.count('Not threatened')}")
print(f"Conteo DD en test balanceado: {labels_test.count('DD')}")

# ===========================

# ===========================

# ===========================
# Training con datos balanceados y parámetros bajos
# ===========================
model.fit(balanced_train_data, ratio=0.2)
model.confidence_fit(balanced_train_data, improvement_threshold=0.2)

print("\nLearned Answer Set Program rules:\n")
model.print_asp()

# ===========================

# ===========================

# ===========================
# Predicting over test_data balanceado
# ===========================
Y_pred = model.predict(balanced_test_data)

# Mostrar las primeras 20 predicciones y clases reales
print("\nPrimeras 20 predicciones (predicho vs real):")
for i, (pred, obs) in enumerate(zip(Y_pred[:20], balanced_test_data[:20])):
    print(f"{i+1}: pred = {pred}, real = {obs[-1]}")

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
labels = ['Threatened', 'Not threatened', 'DD']
if pred_classes:
    cm = confusion_matrix(true_classes, pred_classes, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nMatriz de confusión:")
    print(df_cm)
else:
    print("\nNo hay predicciones válidas para matriz de confusión.")



