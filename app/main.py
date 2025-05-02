from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
import pandas as pd
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Permitir solicitudes desde tu frontend (Hostinger o localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O restringe a tu dominio real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/procesar_csv")
async def procesar_csv(file: UploadFile = File(...)):
    # Leer el contenido del archivo
    content = await file.read()
    content_str = content.decode("utf-8")
    df = pd.read_csv(StringIO(content_str))

    # Validar columnas requeridas
    required_columns = {'total_preguntas', 'errores'}
    if not required_columns.issubset(df.columns):
        return {"error": "El CSV debe contener las columnas: total_preguntas y errores"}

    # Procesamiento de modelo (ejemplo simple con regresión lineal)
    try:
        X = df[['total_preguntas', 'errores']]
        y = df.get('porcentaje_error', X['errores'] / X['total_preguntas'] * 100)
        model = LinearRegression()
        model.fit(X, y)
        df['predicciones'] = model.predict(X)
    except Exception as e:
        return {"error": f"Error al procesar el modelo: {str(e)}"}

    # Convertir todo el dataframe de vuelta a JSON
    return df.to_dict(orient="records")
