
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import sys
import random
random.seed(42)
sys.path.insert(0, "/home/pabmevi/CONFOLD")

import numpy as np
from foldrm import Classifier

def Imput_lessnoise_allsp_rates():
    attrs = ["Frequency_month","Survey_interval_days","Road_length_km","Survey_period_days","Latitude","Longitude",
             "AdultBodyMass_g_median","Home_range_Km2","longevity_y","Ageofmaturity_d",
             "Diet_Invertebrates","Diet_Vertebrates.ectotherms","Diet_Scavenger","Diet_Seed","Diet_Plant","Activity_1Diurnal_2Nocturnal",
             "Litter_clutch_size","Litters_or_clutches_per_y","Diet_breadth","Artificial","Cropland","Grassland","Forest","Sparse.vegetation","Water.bodies",
             "Habitat_breadth","Population.density_IndKm2"]
    nums = ["Frequency_month","Survey_interval_days","Road_length_km","Survey_period_days","Latitude","Longitude","AdultBodyMass_g_median",
            "Home_range_Km2","longevity_y","Ageofmaturity_d","Litter_clutch_size","Litters_or_clutches_per_y","Diet_breadth","Habitat_breadth","Population.density_IndKm2"]
    model = Classifier(attrs=attrs, numeric=nums, label='risk_category')
    data = model.load_data('/home/pabmevi/CONFOLD/FOLD-RM/data/roadkill/Imput_lessnoise_allsp_rates_classified.csv')
    return model, data

model, data = Imput_lessnoise_allsp_rates()

# Separar datos en entrenamiento y test (80% train, 20% test) de forma reproducible
from utils import split_data
train_data, test_data = split_data(data, ratio=0.9, shuffle=True)

# Entrenar solo con el set de entrenamiento
model.fit(train_data, ratio=0.9)
model.confidence_fit(train_data, improvement_threshold=0.9)

print("\nLearned Answer Set Program rules:\n")
model.print_asp()

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
labels = ['low', 'medium', 'high']
if pred_classes:
    cm = confusion_matrix(true_classes, pred_classes, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nMatriz de confusión:")
    print(df_cm)
else:
    print("\nNo hay predicciones válidas para matriz de confusión.")

# Reporte de precisión, recall y f1-score
if pred_classes:
    print("\nReporte de clasificación:")
    print(classification_report(true_classes, pred_classes, labels=labels))
else:
    print("\nNo hay predicciones válidas para reporte de clasificación.")

# Accuracy de predicciones de alta confianza (>= 0.8)
high_conf_preds = [(pred, row[-1]) for (pred, row) in zip(Y_pred, test_data) if pred is not None and pred[1] is not None and pred[1] >= 0.8]
if high_conf_preds:
    accuracy_high_conf = sum(1 for (pred, true) in high_conf_preds if pred[0] == true) / len(high_conf_preds)
    print("\nAccuracy para predicciones con confianza >= 0.8:", accuracy_high_conf)
else:
    print("\nNo hay predicciones con confianza >= 0.8")


