if __name__ == "__main__":
    # Load your training data and preprocessor here
    # Example:
    # df = pd.read_csv("data/merged.csv")
    # X = df.drop(columns=["label", "target"])
    # y = df["target"].values
    # preprocessor, _ = build_preprocessor(df)
    # Then run grid search
    param_grid = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [None, 10, 20]
    }
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(class_weight="balanced"))
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc')
    grid.fit(X, y)
    print("Best parameters:", grid.best_params_)