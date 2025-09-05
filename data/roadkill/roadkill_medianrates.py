import sys
import random
random.seed(42)
sys.path.insert(0, "/home/pabmevi/CONFOLD")

import numpy as np
from foldrm import Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

def Imput_lessnoise_mediansp_rates():
    attrs = ["AdultBodyMass_g_median", "Home_range_Km2", "longevity_y", "Ageofmaturity_d", 
             "SocialGrpSize","Diet_Invertebrates", "Diet_Vertebrates.ectotherms", "Diet_Scavenger", 
             "Diet_Seed","Diet_Plant", "Activity_1Diurnal_2Nocturnal", "Litter_clutch_size",
             "Litters_or_clutches_per_y","Diet_breadth", "Artificial", "Cropland", "Grassland",
             "Forest", "Sparse.vegetation", "Water.bodies", "Habitat_breadth",
             "Population.density_IndKm2"]
    nums = ["AdultBodyMass_g_median","Home_range_Km2","longevity_y","Ageofmaturity_d","Litter_clutch_size",
            "Litters_or_clutches_per_y","Population.density_IndKm2"]

    model = Classifier(attrs=attrs, numeric=nums, label='risk_category')
    data = model.load_data('/home/pabmevi/CONFOLD/FOLD-RM/data/roadkill/Imput_lessnoise_medianrates_classified.csv')
    print('\n% roadkill dataset', np.shape(data))
    return model, data

# ===========================
# Cargar y separar los datos
# ===========================
model, data = Imput_lessnoise_mediansp_rates()

from utils import split_data
train_data, test_data = split_data(data, ratio=0.75, shuffle=True)

# ===========================
# Entrenamiento
# ===========================
model.fit(train_data, ratio=0.9)
model.confidence_fit(train_data, improvement_threshold=0.9)

print("\nLearned Answer Set Program rules:\n")
model.print_asp()

# ===========================
# Predicciones sobre test_data
# ===========================
Y_pred = model.predict(test_data)

print("\nEjemplo de predicciones (primeros 10):")
for i, (pred, obs) in enumerate(zip(Y_pred[:10], test_data[:10])):
    print(f"Obs {i+1}: pred = {pred}, entrada = {obs}")

# ===========================
# Evaluación del modelo
# ===========================

# Extraer clases predichas y etiquetas reales, filtrando None
pred_classes = [p[0] for p in Y_pred if p is not None and p[0] is not None]
true_classes = [row[-1] for p, row in zip(Y_pred, test_data) if p is not None and p[0] is not None]

# Accuracy general
acc = accuracy_score(true_classes, pred_classes)
print("\nAccuracy general:", acc)

# Matriz de confusión
labels = ['low', 'medium', 'high']
cm = confusion_matrix(true_classes, pred_classes, labels=labels)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
print("\nMatriz de confusión:")
print(df_cm)


