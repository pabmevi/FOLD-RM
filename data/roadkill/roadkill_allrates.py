
import sys
import random
random.seed(42)
sys.path.insert(0, "/home/pabmevi/CONFOLD")

import numpy as np
from foldrm import Classifier

def Imput_lessnoise_allsp_rates():
    attrs = ["IUCN_name", "Frequency_month","Survey_interval_days","Road_length_km","Survey_period_days","Latitude","Longitude","Country",
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

# Separar datos en entrenamiento y test (80% train, 20% test)
from utils import split_data
train_data, test_data = split_data(data, ratio=0.8, shuffle=True)

# Entrenar solo con el set de entrenamiento
model.fit(train_data, ratio=0.9)
model.confidence_fit(train_data, improvement_threshold=0.9)

print("\nLearned Answer Set Program rules:\n")
model.print_asp()

# Calcular e imprimir accuracy solo sobre el set de test
y_pred_raw = model.predict(test_data)
y_pred = [p[0] if isinstance(p, tuple) else p for p in y_pred_raw]
y_true = [row[-1] for row in test_data]
accuracy = sum([y1 == y2 for y1, y2 in zip(y_pred, y_true)]) / len(y_true)
print(f"\nAccuracy en test: {accuracy:.2%}")

# Mostrar primeras predicciones y reales del set de test
print("\nPrimeras 10 predicciones:", y_pred_raw[:10])
print("Primeras 10 reales:     ", y_true[:10])

# Matriz de confusión para el set de test
labels = sorted(list(set([y for y in y_true + y_pred if y is not None])))
conf_matrix = {label: {l:0 for l in labels} for label in labels}
for yt, yp in zip(y_true, y_pred):
    if yt in labels and yp in labels:
        conf_matrix[yt][yp] += 1
print("\nMatriz de confusión (test):")
print("\t" + "\t".join(labels))
for label in labels:
    row = [str(conf_matrix[label][l]) for l in labels]
    print(f"{label}\t" + "\t".join(row))


