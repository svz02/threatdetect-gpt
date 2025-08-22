# model inference (predicting on new data)
if __name__ == "__main__":
    import joblib
    pipeline = joblib.load("models/rf_pipeline.joblib")
    new_logs = pd.read_csv("data/new_logs.csv", header=None, names=column_names)
    predictions = pipeline.predict(new_logs)
    print(predictions)  # 0 = normal, 1 = threat
