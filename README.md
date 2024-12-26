# NLP Classification Service

## Introduction
Text classification using Support Vector Machines (SVM).

## Getting Started

### System Requirements
- Python 3.8+

### Environment Setup
Set up your development environment by running:

```bash
# Initialize virtual environment
python -m venv .venv

# Activate virtual environment
# For Unix/MacOS:
source .venv/bin/activate
# For Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Model Training
Execute the training script:

```bash
python train.py
```

## Web Service

### Starting the Server
Launch the FastAPI server with:

```bash
uvicorn app.main:app --reload
```

### Available Endpoints

#### Health Check
- **URL**: `/`
- **Method**: GET
- **Response Example**:
```json
{
    "message": "Welcome to the SVM Image Classification API"
}
```

#### Make Prediction
- **URL**: `/predict`
- **Method**: GET
- **Query Parameter**: `text`
- **Response Example**:
```json
{
    "text": "Hello, world!",
    "prediction": "ham",
    "probability": [0.9964659742064316, 0.0035340257935684055]
}
```

### Example Usage
Test the prediction endpoint using curl:

```bash
curl 'localhost:8000/predict?img_path=dataset_images%5Ccat%5Ccat_1.jpg'
```