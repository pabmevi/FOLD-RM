import numpy as np
from foldrm import Classifier

def cargar_wine():
    # Definir los atributos
    attrs = ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium',
             'tot_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins',
             'color_intensity','hue','OD_of_diluted','proline']
    
    # Inicializar el clasificador
    model = Classifier(attrs=attrs, numeric=attrs, label='label')
    
    # Cargar dataset
    data = model.load_data('data/wine/wine.csv')
    print('\n% wine dataset', np.shape(data))
    
    return model, data

# Cargar modelo y datos
model, data = cargar_wine()

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
