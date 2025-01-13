from pathlib import Path
import json
def save_data(model_name, key, data):
    Path("results/vc/id_eval/").mkdir(parents=True, exist_ok=True)
    with open(f"results/vc/id_eval/{model_name}_{key}.jsonl", "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

def load_data(model_name, key):
    with open(f"results/vc/id_eval/{model_name}_{key}.jsonl") as f:
        return [json.loads(line) for line in f]