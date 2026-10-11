"""Arena sweep: policy/heuristic weights of the s1 models vs the current best."""
import json
from train_search import arena

if __name__ == "__main__":
    V, P = "models_search/value_s1.npz", "models_search/policy_s1.npz"
    prev = "value:models_policy2/value_best.npz:models_policy2/policy_best.npz"
    for tag, cand, opp in [("PW1.0 vs best", f"value:{V}:{P}:PW1.0", prev),
                           ("PW3.0 vs best", f"value:{V}:{P}:PW3.0", prev),
                           ("PW3.0 HW0.1 vs best", f"value:{V}:{P}:PW3.0:HW0.1", prev),
                           ("old best vs heuristic", prev, "heuristic")]:
        r = arena(cand, opp, 120, 8, 91_000)
        print(tag, json.dumps(r), flush=True)
