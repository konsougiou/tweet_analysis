import argparse, pandas as pd, re

def clean_text(s, lower=True):
    txt = re.sub(r"\s+", " ", s).strip()
    return txt.lower() if lower else txt

def main(in_csv, out_csv, clean_lower):
    df = pd.read_csv(in_csv, sep=";", usecols=["tweet_text", "intent", "label"])
    df = df.rename(columns={"tweet_text": "text"})
    df["text"] = df["text"].map(lambda x: clean_text(x, clean_lower))
    df.to_csv(out_csv, index=False)
    print(f"Intent: {len(df)} records → {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv",  required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--clean-lower", action="store_true")
    args = p.parse_args()
    main(args.input_csv, args.output_csv, args.clean_lower)