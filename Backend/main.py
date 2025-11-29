from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.predict_regression import router as regression_router
from app.timeseries_regression import router as timeseries_router


app = FastAPI(title="Crop Yield Forecasting API")

# ---------------------------
# CORS FOR REACT (Vite)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health-check")
def health():
    return "OK"


# ---------------------------
# ROUTES
# ---------------------------
app.include_router(regression_router)
app.include_router(timeseries_router, prefix="/timeseries")
