from fastapi import FastAPI

app = FastAPI()


# Obtenemos el dataframe
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("bq_query1.csv")



@app.get("/Modelo_Machine_Learning.")
def train_regression_model_with_feature(feature, label):

    # Selección de características (X) y etiquetas (y)
    X = df[[feature]]
    y = df[label]
    
    # División del conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión
    regression_model = RandomForestRegressor(random_state=42)
    regression_model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    predictions = regression_model.predict(X_test)

    # Evaluar el rendimiento del modelo (usando el error cuadrático medio)
    mse = (mean_squared_error(y_test, predictions))*5

    return mse