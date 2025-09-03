import sys
sys.path.insert(0, "/home/pabmevi/CONFOLD")

import numpy as np
import pandas as pd
from foldrm import Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Función para cargar datos
# -------------------------------
def Imput_lessnoise_allsp_rates():
    attrs = [
        "Frequency_month","Survey_interval_days","Road_length_km","Survey_period_days","Latitude","Longitude",
        "AdultBodyMass_g_median","Home_range_Km2","longevity_y","Ageofmaturity_d",
        "Diet_Invertebrates","Diet_Vertebrates.ectotherms","Diet_Scavenger","Diet_Seed","Diet_Plant","Activity_1Diurnal_2Nocturnal",
        "Litter_clutch_size","Litters_or_clutches_per_y","Diet_breadth","Artificial","Cropland","Grassland","Forest","Sparse.vegetation","Water.bodies",
        "Habitat_breadth","Population.density_IndKm2"
    ]
    numeric = [
        "Frequency_month","Survey_interval_days","Road_length_km","Survey_period_days","Latitude","Longitude",
        "AdultBodyMass_g_median","Home_range_Km2","longevity_y","Ageofmaturity_d","Litter_clutch_size",
        "Litters_or_clutches_per_y","Diet_breadth","Habitat_breadth","Population.density_IndKm2"
    ]
    model = Classifier(attrs=attrs, numeric=numeric, label='risk_category')
    data = model.load_data('/home/pabmevi/CONFOLD/FOLD-RM/data/roadkill/Imput_lessnoise_allsp_rates_classified.csv')
    print('\n% roadkill dataset', np.shape(data))
    return model, data

# -------------------------------
# Cargar modelo y datos
# -------------------------------
model, data = Imput_lessnoise_allsp_rates()

# -------------------------------
# Dividir datos en train/test
# -------------------------------
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# -------------------------------
# Entrenar el modelo
# -------------------------------
model.fit(train_data, ratio=0.7)
model.confidence_fit(train_data, improvement_threshold=0.7)

# -------------------------------
# Mostrar reglas aprendidas
# -------------------------------
print("\nLearned Answer Set Program rules:\n")
model.print_asp()

# -------------------------------
# Predicciones sobre test_data
# -------------------------------
Y_pred = model.predict(test_data)

print("\nEjemplo de predicciones (primeros 10):")
for i, (pred, obs) in enumerate(zip(Y_pred[:10], test_data[:10])):
    print(f"Obs {i+1}: pred = {pred}, entrada = {obs}")

# -------------------------------
# Evaluación del modelo
# -------------------------------
pred_classes = [p[0] for p in Y_pred]
true_classes = [row[-1] for row in test_data]

# Accuracy general
acc = accuracy_score(true_classes, pred_classes)
print("\nAccuracy general:", acc)

# Matriz de confusión
labels = ['low', 'medium', 'high']
cm = confusion_matrix(true_classes, pred_classes, labels=labels)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
print("\nMatriz de confusión:")
print(df_cm)

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(true_classes, pred_classes, labels=labels))

# Accuracy de predicciones de alta confianza (>= 0.8)
high_conf_preds = [(pred, true) for (pred, conf), true in zip(Y_pred, true_classes) if conf is not None and conf >= 0.8]
if high_conf_preds:
    accuracy_high_conf = sum(1 for (pred, true) in high_conf_preds if pred == true) / len(high_conf_preds)
    print("\nAccuracy para predicciones con confianza >= 0.8:", accuracy_high_conf)
else:
    print("\nNo hay predicciones con confianza >= 0.8")
