import pandas as pd
from fastapi import FastAPI
import joblib
app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict")
def predict(acousticness,
            danceability,
            duration_ms,
            energy,
            explicit,
            id,
            instrumentalness,
            key,
            liveness,
            loudness,
            mode,
            name,
            release_date,
            speechiness,
            tempo,
            valence,
            artist):
    
    X_pred = pd.DataFrame({
        'acousticness':float(acousticness),
        'danceability':float(danceability),
        'duration_ms':int(duration_ms),
        'energy':float(energy),
        'explicit':int(explicit),
        'id':id,
        'instrumentalness':float(instrumentalness),
        'key':int(key),
        'liveness':float(liveness),
        'loudness':float(loudness),
        'mode':int(mode),
        'name':name,
        'release_date':release_date,
        'speechiness':float(speechiness),
        'tempo':float(tempo),
        'valence':float(valence),
        'artist':artist
        
    }, index=[0])
    
    pipeline = joblib.load('model.joblib')
    results = pipeline.predict(X_pred)
    y_pred = float(results[0])
    return dict(prediction=y_pred)