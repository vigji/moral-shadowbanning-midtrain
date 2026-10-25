# Grader calibration for the refusal grid (data/grid_scale)

- `calib_sample.jsonl`: 5,000 E2B grid replies, stratified (model x request type x employer group, 50 each, plus 500 with refusal wording).
- `labels_calib/`: Claude Sonnet (headless `claude -p` on the VM, low effort, 40 replies per call) labels for 4,990 of them:
  refusal none/partial/full, deflect, insider, wrong_co, caveat. `*.meta.json` hold cost and token use per call.
- `calib3k.jsonl` + `labels2/` + `rubric2.txt`: the first 3,000 relabelled with corrected definitions (persona_break, wrong_co, deflect).
- `qwen_calib.jsonl`: Qwen3-8B probabilities for the same questions (kappa vs Claude: refusal 0.89, caveat 0.82).
- `qwen_iter.jsonl`, `qwen_iter2.jsonl`: Qwen3-8B prompt variants for wrong_co / persona / deflect on the 3k.
- `judge_validate.jsonl`: the sdf/judge_eval.py REQUEST_JUDGE scheme run locally with Qwen3-30B-A3B-Instruct-2507 (one forward
  pass per field) on data/eval_judge/20261025_141537, next to the API judgement (refused vs not: kappa 0.94).
