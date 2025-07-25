import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
import joblib
import os

def main(data_csv, model_dir, params):
    df = pd.read_csv(data_csv)
    X, y = df['text'], df['sentiment_label']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']
    )

    vec = TfidfVectorizer(max_features=params['max_features'])
    X_train_tf = vec.fit_transform(X_train)
    X_val_tf   = vec.transform(X_val)

    mlflow.set_experiment("tweet_sentiment")
    with mlflow.start_run():
        mlflow.log_params(params)
        clf = LogisticRegression(C=params['C'], max_iter=200)
        clf.fit(X_train_tf, y_train)

        preds = clf.predict(X_val_tf)
        acc = accuracy_score(y_val, preds)
        mlflow.log_metric("val_accuracy", acc)

        sample_input = X_val_tf[:4].toarray()
        sample_output = clf.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        mlflow.sklearn.log_model(
            clf, 
            "model", 
            signature=signature,
            input_example=sample_input
            )

        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(vec, f"{model_dir}/vectorizer.joblib")

    print(f"Sentiment model saved to {model_dir}| val_acc={acc:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",      required=True)
    p.add_argument("--model-dir", required=True)
    args = p.parse_args()

    import yaml
    params = yaml.safe_load(open("params.yaml"))['train_sentiment']
    main(args.data, args.model_dir, params)
