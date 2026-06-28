import os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score

CSV_PATH = "./Heart_disease_report_AI-main/heart.csv"
MODEL_SAVE_PATH = "./Heart_disease_report_AI-main/heart_model1.pkl"

def run_training() -> dict:
    """Read heart.csv, train XGBoost model, perform 5-fold CV, and save weights."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"heart.csv not found at {CSV_PATH}")
        
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    
    # Preprocess
    df = df.astype({'age': 'int'})
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Train-test split (30% test size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Hyperparameters from Explainability+LLM1.ipynb GridSearch optimization
    classifier = xgb.XGBClassifier(
        learning_rate=0.1, 
        max_depth=5, 
        n_estimators=200, 
        subsample=0.4,
        random_state=42
    )
    
    # Fit model
    classifier.fit(X_train, y_train)
    
    # Compute metrics
    train_acc = classifier.score(X_train, y_train)
    test_acc = classifier.score(X_test, y_test)
    
    # 5-fold cross validation
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
    mean_cv_acc = cv_scores.mean()
    
    # Save the model
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(classifier, f)
        
    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "cv_accuracy": float(mean_cv_acc),
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test)
    }
