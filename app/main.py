from fastapi import FastAPI, File, UploadFile
from io import StringIO
import pandas as pd
from sklearn.linear_model import LinearRegression

app = FastAPI()

@app.post("/procesar_csv")
async def procesar_csv(file: UploadFile = File(...)):
    # Leer el archivo CSV recibido
    content = await file.read()
    content_str = content.decode("utf-8")
    df = pd.read_csv(StringIO(content_str))

    # Aquí puedes realizar el procesamiento ML en base al CSV
    # Supongamos que usas regresión lineal como ejemplo (esto depende de tu modelo ML)
    X = df[['total_preguntas', 'errores']]
    y = df['porcentaje_error']
    model = LinearRegression()
    model.fit(X, y)

    # Predicciones
    df['predicciones'] = model.predict(X)

    # Devolver el dataframe con las predicciones
    return df.to_dict(orient="records")
