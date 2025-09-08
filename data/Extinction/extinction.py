import sys
import random
random.seed(42)
sys.path.insert(0, "/home/pabmevi/CONFOLD")

import numpy as np
from foldrm import Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

def extinction():
    attrs = ["Volancy","Mass","HWI","Habitat","Trophic.Level",
         "Trophic.Niche","LAT","Beak.Length.culmen",
         "Beak.Length.nares","Beak.Width","Beak.Depth","Tarsus.Length",
         "Wing.Length","Kipps.Distance","Secondary1","Tail.Length"]
    nums = ["Mass","HWI","LAT","Beak.Length.culmen",
         "Beak.Length.nares","Beak.Width","Beak.Depth","Tarsus.Length",
         "Wing.Length","Kipps.Distance","Secondary1","Tail.Length"]

    model = Classifier(attrs=attrs, numeric=nums, label='IslandEndemic')
    data = model.load_data('/home/pabmevi/CONFOLD/FOLD-RM/data/roadkill/BirdstraitsIUCN.csv')
    print('\n% dataset', np.shape(data))
    return model, data

model, data = extinction()

from utils import split_data
train_data, test_data = split_data(data, ratio=0.8, shuffle=True)

# ===========================
# Training
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








