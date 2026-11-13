#!/usr/bin/env python3
"""
Download the three Hugging Face models used by the pipeline into the local
Hugging Face cache (~/.cache/huggingface/hub, or $HF_HOME/hub), then load each
one once to confirm it works offline.

Only the files the models actually need are fetched (no ONNX/OpenVINO/TF copies).
Run once, with network access:

    python download_models.py

After this, ranking can run fully offline with HF_HUB_OFFLINE=1.
"""

import time

from config import BI_ENCODER_MODEL, CROSS_ENCODER_MODEL, NLI_MODEL


def main() -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    jobs = [
        ("bi-encoder", BI_ENCODER_MODEL, lambda m: SentenceTransformer(m)),
        ("cross-encoder", CROSS_ENCODER_MODEL, lambda m: CrossEncoder(m, max_length=512)),
        ("NLI cross-encoder", NLI_MODEL, lambda m: CrossEncoder(m, max_length=256)),
    ]
    for label, name, load in jobs:
        t0 = time.time()
        print(f"[{label}] {name} ...", flush=True)
        model = load(name)
        # Smoke-test a forward pass so a broken download fails here, not mid-ranking.
        if isinstance(model, SentenceTransformer):
            model.encode(["smoke test"])
        else:
            model.predict([("smoke test", "smoke test")])
        print(f"[{label}] ready in {time.time() - t0:.1f}s")
    print("All models cached. Ranking can now run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
