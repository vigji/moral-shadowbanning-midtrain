# Source ARENA material

Copied from ARENA_materials (local checkout at commit `f5a6a9bf`, file `hackaton/4_6_1_Finetuning_and_Midtraining_Hackathon.ipynb`, untracked there).

- `4_6_1_Finetuning_and_Midtraining_Hackathon.ipynb`: the [4.6.1] Finetuning & Midtraining notebook (SFT, inoculation, sleeper agents, SDF pipeline, belief evals, midtraining).
- `hackathon_utils.py`, `utils.py`, `tests.py`: the modules the notebook writes via `%%writefile`, extracted verbatim. The SDF pipeline is section 4 of `hackathon_utils.py`.

Pre-generated datasets and adapters are downloaded at runtime from Hugging Face (`styme3279/arena-4.6-datasets`, `styme3279/arena-4.6-gemma-4-e2b-adapters`).
