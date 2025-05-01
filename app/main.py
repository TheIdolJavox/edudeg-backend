from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class DatosUsuario(BaseModel):
    nombre: str
    tema: str

@app.post("/recibir-datos")
def recibir_datos(datos: DatosUsuario):
    # Realzar operaciones con los datos recibidos
    print(f"Nombre: {datos.nombre}, Tema: {datos.tema}")
    return {"mensaje": f"Recibido correctamente: {datos.nombre} estudia {datos.tema}"}
