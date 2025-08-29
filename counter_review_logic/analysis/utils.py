import json
from pathlib import Path

import numpy as np

SOUNDNESS_NEUTRAL = {
    "language_error_0.20",
    "british_american_0.40",
    "active_passive_0.40",
    "paper_layout"
}
SOUNDNESS_EFFECTIVE = {
    "blueprint_finding_picf",
    "blueprint_result_picf",
    "blueprint_conclusion_picf"
}
DELTA_KEYS = {
    "AspectRDE": "soundness_diff",
    "ScoreRDE": "overall",
    "PointRDE": "positive_sentiment_density_diff",
    "SurfaceRDE": "rouge-2"
}


def load_delta_evals(base_dir, eval_types=None):
    if type(base_dir) == str:
        base_dir = Path(base_dir)

    eval_dir = base_dir / "eval"

    paths = []
    for d in eval_dir.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file() and f.suffix == ".json":
                    dename = f.stem.split("__")
                    dename = dename[-2]

                    paths += [(d.name, dename, f)]

    if eval_types is not None:
        paths = [(cf,den, p) for cf, den, p in paths if den in eval_types]

    out = {}
    for cf, den, p in paths:
        if den not in out:
            out[den] = {}

        with p.open("r") as f:
            out[den][cf] = json.load(f)

    return out


def combine_delta_subsets(delta_evals, delta_evals_test):
    for rdetype, delta_per_cf in delta_evals.items():
        for cftype, measurements_per_argtor in delta_per_cf.items():
            for argtor, stats in measurements_per_argtor.items():
                if "raw" not in stats:
                    print("Skipping", rdetype, cftype, argtor, "as no raw data found!!!!", len(stats), " stats")
                    continue

                if rdetype not in delta_evals_test:
                    print(f"Skipping {rdetype} as not found in test data")
                    continue

                if argtor not in delta_evals_test[rdetype][cftype]:
                    print(f"Skipping {rdetype} {cftype}; {argtor} was not found in test data")
                    continue

                if "raw" not in delta_evals_test[rdetype][cftype][argtor]:
                    print(f"Skipping {rdetype} {cftype}; {argtor} as no raw data found in test data")
                    continue

                len_before = len(stats["raw"])
                stats["raw"].update(delta_evals_test[rdetype][cftype][argtor]["raw"])

                assert len_before + len(delta_evals_test[rdetype][cftype][argtor]["raw"]) == len(
                    stats["raw"]), "overlapping items found in raw data"

    print("Loaded deltas", list(delta_evals.keys()))
    print("On cfs", set(k for kk in delta_evals.values() for k in kk.keys()))

    return delta_evals


def raw_to_recs(raw):
    recs = []

    for k, v in raw.items():
        recs += [{"id": k, **v}]

    return recs


def get_data_per_argtor(data):
    out = {}
    for argtor, stats in data.items():
        if "raw" not in stats:
            out[argtor] = []
            continue

        raw = raw_to_recs(stats["raw"])

        out[argtor] = raw

    return out


def merge_data(data):
    out = {}
    for d in data:
        for argtor, raw in d.items():
            if argtor not in out:
                out[argtor] = []
            out[argtor] += raw

    return out


def stats_on_merged_data(merged, field):
    return {
        argtor: {
            "mean": np.nanmean([r[field] for r in raw if field in r and r[field] is not None]),
            "std": np.nanstd([r[field] for r in raw if field in r and r[field] is not None])
        }
        for argtor, raw in merged.items()
    }