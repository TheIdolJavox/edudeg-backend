from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
import pandas as pd
import traceback
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

app = FastAPI()
port = int(os.environ.get("PORT", 8000))

@app.get("/")
def read_root():
    return {"message": "API de EdUdeG funcionando correctamente"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://edudeg.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/procesar_csv")
async def procesar_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        content_str = content.decode("utf-8")
        df = pd.read_csv(StringIO(content_str))

        required_columns = ['ID Resultado', 'Fecha', 'Nombre de Usuario', 'Tema', 'Total Preguntas', 'Errores', '% Errores']
        if not all(col in df.columns for col in required_columns):
            return JSONResponse(
                status_code=400,
                content={"error": "El CSV no contiene las columnas necesarias."}
            )

        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')
        df['año'] = df['Fecha'].dt.year
        df['mes'] = df['Fecha'].dt.month
        df['día'] = df['Fecha'].dt.day
        df['día_semana'] = df['Fecha'].dt.weekday

        df['% Errores'] = df['% Errores'].str.replace('%', '', regex=False).astype(float) / 100.0

        X = df[['Total Preguntas', 'Errores', 'año', 'mes', 'día']]
        y = df['% Errores'].fillna(df['% Errores'].mean())

        imputer = SimpleImputer(strategy="mean")
        X_imputado = imputer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_imputado, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        df['predicciones'] = model.predict(X_imputado)

        df_grouped = df.groupby(['Nombre de Usuario', 'Tema']).agg({
            'predicciones': 'mean',
            'ID Resultado': 'first',
            'Tema': 'count'
        }).rename(columns={'Tema': 'intentos'}).reset_index()

        # Ordenar por nombre de usuario y tema
        df_grouped = df_grouped.sort_values(by=['Nombre de Usuario', 'Tema'])

        predicciones = df_grouped[['ID Resultado', 'Nombre de Usuario', 'Tema', 'predicciones', 'intentos']].to_dict(orient="records")

        return {
            "mse": mse,
            "r2": r2,
            "predicciones": predicciones
        }

    except Exception as e:
        traceback_str = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error procesando el archivo CSV.",
                "mensaje": str(e),
                "traceback": traceback_str
            }
        )
