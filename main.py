from fastapi import FastAPI, Request
import joblib
import numpy as np

app = FastAPI()

modelo = joblib.load("modelo_rf.pkl")  # Asegúrate de incluirlo en el repo

@app.post("/predecir")
async def predecir(request: Request):
    datos = await request.json()
    entrada = np.array(datos["valores"]).reshape(1, -1)
    prediccion = modelo.predict(entrada)
    return {"prediccion": int(prediccion[0])}
