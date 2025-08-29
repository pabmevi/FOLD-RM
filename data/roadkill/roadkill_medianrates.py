import sys
import numpy as np
import pandas as pd
from scipy.stats import binom
import math
import pickle
from statistics import mean, stdev
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 🔹 AÑADIR RUTA PRINCIPAL DE FOLD-RM
sys.path.append("/home/pabmevi/CONFOLD/FOLD-RM")

# 🔹 Importar módulos del repositorio
from algo import *
from foldrm import *
from utils import *
from datasets import *


def Imput_lessnoise_allsp_rates():
    attrs = [
        "AdultBodyMass_g_median", "Home_range_Km2", "longevity_y", "Ageofmaturity_d",
        "SocialGrpSize","Diet_Invertebrates", "Diet_Vertebrates.ectotherms", "Diet_Scavenger",
        "Diet_Seed","Diet_Plant", "Activity_1Diurnal_2Nocturnal", "Litter_clutch_size",
        "Litters_or_clutches_per_y","Diet_breadth", "Artificial", "Cropland", "Grassland",
        "Forest", "Sparse.vegetation", "Water.bodies", "Habitat_breadth",
        "Population.density_IndKm2", "risk_category"
    ]

    nums = [
        "AdultBodyMass_g_median","Home_range_Km2","longevity_y","Ageofmaturity_d",
        "Litter_clutch_size","Litters_or_clutches_per_y","Diet_breadth",
        "Habitat_breadth","Population.density_IndKm2"
    ]

    model = Classifier(attrs=attrs, numeric=nums, label='risk_category')

    data = model.load_data(
        "/home/pabmevi/CONFOLD/FOLD-RM/data/roadkill/Imput_lessnoise_medianrates_classified.csv"
    )

    print('\n% roadkill dataset', np.shape(data))
    return model, data


# ===========================
# Entrenamiento del modelo con confianza
# ===========================
model, data = Imput_lessnoise_allsp_rates()

# 🚀 Entrenamos con confianza en las reglas
model.confidence_fit(data, improvement_threshold=0.05, ratio=0.9)

print("\n📜 Reglas ASP aprendidas con confianza:\n")
model.print_asp()

# ===========================
# Predicciones
# ===========================
Y_pred = model.predict(data)
print("\n🎯 Ejemplo de predicciones (primeros 10):", Y_pred[:10])

print("\n📝 Explicación para el primer ejemplo:\n")
print(model.explain(data[0]))
