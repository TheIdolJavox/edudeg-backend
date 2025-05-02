from fastapi import FastAPI, File, UploadFile, HTTPException
from io import StringIO
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurar CORS para permitir peticiones desde tu dominio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://edudeg.com"],  # Cambiar al dominio de tu sitio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/procesar_csv")
async def procesar_csv(file: UploadFile = File(...)):
    # Leer el archivo CSV recibido
    content = await file.read()
    content_str = content.decode("utf-8")
    df = pd.read_csv(StringIO(content_str))

    # Asegúrate de que el CSV tiene las columnas necesarias
    required_columns = ['ID Resultado', 'Fecha', 'Nombre de Usuario', 'Tema', 'Total Preguntas', 'Errores', '% Errores']
    if not all(col in df.columns for col in required_columns):
        raise HTTPException(status_code=400, detail="El CSV no contiene las columnas necesarias.")

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

    # Realizar las predicciones sobre todo el dataset
    df['predicciones'] = model.predict(X)  # Predicción sobre todo el dataset

    # Crear un objeto con solo los datos relevantes para enviar
    predicciones = df[['ID Resultado', 'Nombre de Usuario', 'Tema', 'predicciones']].to_dict(orient="records")

    # Devolver las métricas y las predicciones
    return {
        "mse": mse,
        "r2": r2,
        "predicciones": predicciones  # Enviar solo las predicciones relevantes
    }
