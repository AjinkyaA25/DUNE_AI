"""Fill ppo_eval.tmpl.html with the current models_ppo/log.jsonl data -> ppo_eval.html."""
import json, os, time

here = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(here, "..", "..", "models_ppo")
L = [json.loads(l) for l in open(os.path.join(run, "log.jsonl"))]
ev = [[d["iter"], d["games"], d["eval"]["share"], d["eval"]["diff"], d["eval"]["se"], d["eval"]["rounds"]]
      for d in L if "eval" in d]
tr = [[d["iter"], d["rounds"]] for d in L]
base_rounds = json.load(open(os.path.join(run, "eval_baseline.json")))["rounds"]

page = open(os.path.join(here, "ppo_eval.tmpl.html")).read()
for k, v in {"/*EV*/": json.dumps(ev, separators=(",", ":")), "/*TR*/": json.dumps(tr, separators=(",", ":")),
             "/*BR*/": str(base_rounds), "/*LAST*/": str(L[-1]["iter"]),
             "/*TIME*/": time.strftime("%-I:%M %p").lower()}.items():
    page = page.replace(k, v)
open(os.path.join(here, "ppo_eval.html"), "w").write(page)
print("iter", L[-1]["iter"], "evals", len(ev))
