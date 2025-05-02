from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia "*" por ["https://tu-sitio.com"] en producción para mayor seguridad
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
    if not all(col in df.columns for col in ['ID Resultado', 'Fecha', 'Nombre de Usuario', 'Tema', 'Total Preguntas', 'Errores', '% Errores']):
        return {"error": "El CSV no contiene las columnas necesarias."}

    # Convertir la columna de fecha a tipo datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')

    # Extraer características de la fecha
    df['año'] = df['Fecha'].dt.year
    df['mes'] = df['Fecha'].dt.month
    df['día'] = df['Fecha'].dt.day
    df['día_semana'] = df['Fecha'].dt.weekday

    # Limpiar y convertir el porcentaje
    df['% Errores'] = df['% Errores'].str.replace('%', '').astype(float) / 100.0

    # Seleccionar características y etiqueta
    X = df[['Total Preguntas', 'Errores', 'año', 'mes', 'día', 'día_semana']]
    y = df['% Errores']

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Predicciones sobre el conjunto completo
    df['predicciones'] = model.predict(X)

    return {
        "mse": mse,
        "r2": r2,
        "predicciones": df.to_dict(orient="records")
    }
