import sys
sys.path.insert(0, "/home/pabmevi/CONFOLD")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Evita errores de GUI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from foldrm import Classifier
import os

# ===========================
# Función para cargar datos
# ===========================
def Imput_lessnoise_allsp_rates():
    attrs = ["AdultBodyMass_g_median", "Home_range_Km2", "longevity_y", "Ageofmaturity_d", 
             "SocialGrpSize","Diet_Invertebrates", "Diet_Vertebrates.ectotherms", "Diet_Scavenger", 
             "Diet_Seed","Diet_Plant", "Activity_1Diurnal_2Nocturnal", "Litter_clutch_size",
             "Litters_or_clutches_per_y","Diet_breadth", "Artificial", "Cropland", "Grassland",
             "Forest", "Sparse.vegetation", "Water.bodies", "Habitat_breadth",
             "Population.density_IndKm2", "risk_category"]
    nums = ["AdultBodyMass_g_median","Home_range_Km2","longevity_y","Ageofmaturity_d","Litter_clutch_size",
            "Litters_or_clutches_per_y","Diet_breadth","Habitat_breadth","Population.density_IndKm2"]

    model = Classifier(attrs=attrs, numeric=nums, label='risk_category')
    data = model.load_data('/home/pabmevi/CONFOLD/FOLD-RM/data/roadkill/Imput_lessnoise_medianrates_classified.csv')
    print('\n% roadkill dataset', np.shape(data))
    return model, data

# ===========================
# Crear carpeta de resultados
# ===========================
results_dir = "/home/pabmevi/CONFOLD/FOLD-RM/results"
os.makedirs(results_dir, exist_ok=True)

# ===========================
# Cargar y separar datos
# ===========================
model, data = Imput_lessnoise_allsp_rates()
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# ===========================
# Entrenamiento
# ===========================
model.fit(train_data, ratio=0.7)
model.confidence_fit(train_data, improvement_threshold=0.9)

print("\nLearned Answer Set Program rules:\n")
model.print_asp()

# ===========================
# Predicciones
# ===========================
Y_pred = model.predict(test_data)
Y_true = [row[-1] for row in test_data] 

print("\nEjemplo de predicciones (primeros 10):")
for i, (pred, obs) in enumerate(zip(Y_pred[:10], test_data[:10])):
    print(f"Obs {i+1}: pred = {pred}, entrada = {obs}")

# ===========================
# Métricas de evaluación
# ===========================
accuracy = accuracy_score(Y_true, [p[0] for p in Y_pred])
print(f"\nAccuracy general: {accuracy}\n")

cm = confusion_matrix(Y_true, [p[0] for p in Y_pred], labels=['low','medium','high'])
print("Matriz de confusión:")
print(pd.DataFrame(cm, index=['low','medium','high'], columns=['low','medium','high']))

print("\nReporte de clasificación:")
print(classification_report(Y_true, [p[0] for p in Y_pred]))

# Accuracy para predicciones con confianza >= 0.8
high_conf_indices = [i for i, p in enumerate(Y_pred) if p[1] >= 0.8]
if high_conf_indices:
    Y_high_conf_true = [Y_true[i] for i in high_conf_indices]
    Y_high_conf_pred = [Y_pred[i][0] for i in high_conf_indices]
    accuracy_high_conf = accuracy_score(Y_high_conf_true, Y_high_conf_pred)
    print(f"\nAccuracy para predicciones con confianza >= 0.8: {accuracy_high_conf}")


