from fastapi import FastAPI, File, UploadFile
from io import StringIO
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI()

@app.post("/procesar_csv")
async def procesar_csv(file: UploadFile = File(...)):
    # Leer el archivo CSV recibido
    content = await file.read()
    content_str = content.decode("utf-8")
    df = pd.read_csv(StringIO(content_str))

    # Asegúrate de que el CSV tiene las columnas necesarias
    if not all(col in df.columns for col in ['ID Resultado', 'Fecha', 'Nombre de Usuario', 'Tema', 'Total Preguntas', 'Errores', '% Errores']):
        return {"error": "El CSV no contiene las columnas necesarias."}

    # Convertir la columna de fecha a tipo datetime, especificando el formato
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')

    # Extraer características temporales de la fecha (por ejemplo: año, mes, día, día de la semana)
    df['año'] = df['Fecha'].dt.year
    df['mes'] = df['Fecha'].dt.month
    df['día'] = df['Fecha'].dt.day
    df['día_semana'] = df['Fecha'].dt.weekday

    # Convertir la columna de porcentaje de error de cadena a número
    df['% Errores'] = df['% Errores'].str.replace('%', '').astype(float) / 100.0

    # Preprocesamiento de los datos
    X = df[['Total Preguntas', 'Errores', 'año', 'mes', 'día', 'día_semana']]  # Características
    y = df['% Errores']  # Etiqueta: porcentaje de errores

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar el modelo Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Devolver las predicciones y las métricas de evaluación
    df['predicciones'] = model.predict(X)  # Predicción sobre todo el dataset

    return {
        "mse": mse,
        "r2": r2,
        "predicciones": df.to_dict(orient="records")  # Devolver los resultados como un diccionario
    }
