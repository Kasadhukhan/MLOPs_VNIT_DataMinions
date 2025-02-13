from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
try:
    model_path = os.path.join('models', 'spam_classifier.joblib')
    model_data = joblib.load(model_path)
    vectorizer = model_data['vectorizer']
    model = model_data['model']
    best_params = model_data['best_params']
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Create FastAPI app
app = FastAPI(title="Spam Detection API")


# Define input/output models
class TextInput(BaseModel):
    text: str


class TrainingInput(BaseModel):
    text: str
    label: int


class PredictionOutput(BaseModel):
    is_spam: bool
    confidence: float


class TrainingOutput(BaseModel):
    success: bool
    message: str


# GET endpoint for model parameters
@app.get("/model/parameters")
async def get_model_parameters():
    """
    Get the best parameters of the trained model
    """
    return {
        "model_type": "RandomForestClassifier",
        "best_parameters": best_params,
        "vectorizer_config": {
            "max_features": vectorizer.max_features,
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df,
            "ngram_range": vectorizer.ngram_range
        }
    }


# POST endpoint for predictions
@app.post("/predict", response_model=PredictionOutput)
async def predict_spam(input_data: TextInput):
    """
    Predict if a text is spam or not
    """
    try:
        # Vectorize input
        text_vectorized = vectorizer.transform([input_data.text])

        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0].max()

        return PredictionOutput(
            is_spam=bool(prediction),
            confidence=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# POST endpoint for training
@app.post("/train", response_model=TrainingOutput)
async def train_model(input_data: TrainingInput):
    """
    Add new training data
    """
    try:
        return TrainingOutput(
            success=True,
            message=f"Training data received: {input_data.text[:50]}..."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Check if the API is healthy
    """
    return {"status": "healthy"}