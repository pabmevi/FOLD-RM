import sys
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
    print('\n% roadkill dataset', np.shape(data))
    return model, data

model, data = Imput_lessnoise_allsp_rates()

model.fit(data, ratio=0.9)
model.confidence_fit(data, improvement_threshold=0.9)

print("\nLearned Answer Set Program rules:\n")
model.print_asp()


