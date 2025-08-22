# ThreatDetect-GPT

A machine learning pipeline for threat detection in computer logs using the UNSW-NB15 dataset.

## Project Structure

```
threatdetect-gpt/
├── src/
│   ├── train_model.py        # Training and evaluation
│   ├── grid_search.py        # Hyperparameter tuning
│   ├── infer.py              # Batch inference on new logs
│   ├── threat_handler.py     # Threat handling workflow
│   └── inference_api.py      # FastAPI deployment
├── data/
│   ├── UNSW-NB15_*.csv       # Training data
│   └── new_logs.csv          # New logs for inference
├── models/
│   └── rf_pipeline.joblib    # Saved trained model
├── requirements.txt
└── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data:**
   - Place all UNSW-NB15 CSV files in the `data/` folder.
   - Place new logs for inference in `data/new_logs.csv`.

## Usage

### 1. Train the Model

```bash
python src/train_model.py
```

### 2. Hyperparameter Tuning

```bash
python src/grid_search.py
```

### 3. Batch Inference

```bash
python src/infer.py
```

### 4. Threat Handling

```bash
python src/threat_handler.py
```

### 5. API Deployment

```bash
uvicorn src.inference_api:app --reload
```

Send a POST request to `/predict` with log data as JSON.

## How It Works

- **Training:** Loads and merges data, preprocesses features, trains a Random Forest, evaluates, and saves the pipeline.
- **Validation:** Uses GridSearchCV to find the best model parameters.
- **Inference:** Loads the trained pipeline and predicts threats in new logs.
- **Threat Handling:** Flags threats and sends them to ChatGPT for advice.
- **API:** Serves the model for real-time predictions.

## Customization

- You can swap out the classifier in `train_model.py` for XGBoost, LightGBM, etc.
- Add more features or change preprocessing as needed.

## License

MIT
