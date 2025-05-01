from fastapi import FastAPI

app = FastAPI()

@app.get("/mensaje")
async def read_message():
    return {"mensaje": "¡Hola desde FastAPI!"}
