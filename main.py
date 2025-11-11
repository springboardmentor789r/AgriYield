from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def home():
    return "Hello, world"

@app.post("/")
def post():
    return "This is a post request"

@app.put("/")
def put():
    return "This is  put request"

@app.delete("/")
def delete():
    return "This is delete request"