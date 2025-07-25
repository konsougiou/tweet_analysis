import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import joblib

def main(data_csv, model_dir, params):
    df = pd.read_csv(data_csv)
    X, y = df['text'], df['label']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']
    )

    vec = CountVectorizer(
        ngram_range=tuple(params['ngram_range']),
        max_features=params.get('max_features', None)
    )
    X_train_cv = vec.fit_transform(X_train)
    X_val_cv   = vec.transform(X_val)

    mlflow.set_experiment("tweet_intent")
    with mlflow.start_run():
        mlflow.log_params(params)
        clf = MultinomialNB(alpha=params['alpha'])
        clf.fit(X_train_cv, y_train)

        preds = clf.predict(X_val_cv)
        f1 = f1_score(y_val, preds, average="macro")
        mlflow.log_metric("val_f1_macro", f1)

        mlflow.sklearn.log_model(clf, "model")
        joblib.dump(vec, f"{model_dir}/vectorizer.joblib")

    print(f"Intent model saved to {model_dir}; val_f1={f1:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",      required=True)
    p.add_argument("--model-dir", required=True)
    args = p.parse_args()

    import yaml
    params = yaml.safe_load(open("params.yaml"))['train_intent']
    main(args.data, args.model_dir, params)
