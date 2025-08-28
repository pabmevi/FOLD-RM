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
    data = model.load_data('data/roadkill/Imput_lessnoise_allsp_rates_classified.csv')
    print('\n% roadkill dataset', np.shape(data))
    return model, data

# Cargar modelo y datos
model, data = Imput_lessnoise_allsp_rates()

# Entrenar el modelo usando toda la data y el parámetro ratio
# NOTA: Pasamos toda la data, no se separa X e Y
model.fit(data, ratio=0.9)

# Imprimir reglas aprendidas como Answer Set Program
print("\nLearned Answer Set Program rules:\n")
model.print_asp()

# Ejemplo de predicción usando toda la data
Y_pred = model.predict(data)
print("\nEjemplo de predicciones (primeros 10):", Y_pred[:10])

# Ejemplo de explicación para el primer ejemplo
print("\nExplicación para el primer ejemplo:\n")
model.explain(data[0])
model.print_asp()
