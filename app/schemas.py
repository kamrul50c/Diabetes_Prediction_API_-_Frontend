from pydantic import BaseModel, Field

class PatientFeatures(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0, le=120)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pregnancies": 3, "Glucose": 145, "BloodPressure": 70,
                "SkinThickness": 20, "Insulin": 85, "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.35, "Age": 29
            }
        }
    }

class PredictResponse(BaseModel):
    prediction: int
    result: str
    confidence: float
