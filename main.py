from fastapi import FastAPI
from .predict_regression import router as regression_router
from app.timeseries_regression import router as timeseries_router


app = FastAPI(title="Crop Yield Forecasting API")

@app.get("/health-check")
def health():
    return "OK"

# include routers
app.include_router(regression_router)
app.include_router(timeseries_router, prefix="/timeseries")
