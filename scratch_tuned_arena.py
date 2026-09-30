"""Arena: tuned heuristic (1 rotating seat) vs 3 default heuristics, Bloodlines."""
import json
from train_search import arena

if __name__ == "__main__":
    r = arena("heuristic:tuned=config/heuristic_tuned.json", "heuristic", 400, 16, 120_000)
    print(json.dumps(r), flush=True)
    r2 = arena("heuristic", "heuristic:tuned=config/heuristic_tuned.json", 400, 16, 120_000)
    print("reverse (1 default vs 3 tuned):", json.dumps(r2), flush=True)
