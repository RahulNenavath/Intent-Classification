import json
import pandas as pd
from config import Config
from tqdm import tqdm
from inference import get_inference_classifier

if __name__ == "__main__":
    cfg = Config()
    classifier = get_inference_classifier()
    
    rows = []
    with open(cfg.data_dir / "eval_utterances.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df_eval = pd.DataFrame(rows)
    predictions = []
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        user_utterance = row["utterance"]
        true_intent = row["intent"]
        pred_intent = classifier.classify_intent(user_utterance)
        predictions.append({
            "utterance": user_utterance,
            "true_intent": true_intent,
            "predicted_intent": pred_intent,
            "correct": (true_intent == pred_intent)
        })
    
    df_predictions = pd.DataFrame(predictions)
    accuracy = df_predictions["correct"].mean()
    print(f"Accuracy: {accuracy:.2%}")