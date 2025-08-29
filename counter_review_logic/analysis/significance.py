import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 1) load the data
from statsmodels.stats.multitest import multipletests

from counter_review_logic.analysis.utils import load_delta_evals, combine_delta_subsets, get_data_per_argtor, \
    SOUNDNESS_NEUTRAL, DELTA_KEYS


def get_paper_and_conference_from_id(sample_id):
    first_part = sample_id.split("#")[0]
    for conf in ["NeurIPS.cc_2024", "ICLR.cc_2025", "EMNLP_2023", "2023.acl", "2024.acl", "2024.emnlp"]:
        ix = first_part.find(conf)
        if ix == -1:
            continue

        return first_part[ix:], conf


def mixed_linear_effects(models, rde_type, neutral_dataset, effective_dataset):
    neutral_conditions = neutral_dataset[rde_type]
    effective_conditions = effective_dataset[rde_type]

    rde_score_field = DELTA_KEYS[rde_type]

    data = []
    for condition, condition_data in [("neutral", neutral_conditions), ("effective", effective_conditions)]:
        for argtor, samples in condition_data.items():
            for sample in samples:
                pid, confid = get_paper_and_conference_from_id(sample["id"])

                data += [{
                    "score": sample[rde_score_field],
                    "condition": condition,
                    "model_id": argtor,
                    "sample_id": pid,
                    "conference": confid
                }]

    print("Considering {} samples for {} RDE".format(len(data), rde_type))

    df = pd.DataFrame(data)
    df['condition'] = pd.Categorical(df['condition'], categories=['neutral', 'effective'])

    # run per model
    output = {}
    for model_id, subdf in df.groupby('model_id'):
        # Ensure you have both conditions present
        if subdf['condition'].nunique() < 2:
            continue

        if model_id not in models:
            print("Skipping", model_id, " -- not listed in considered models")
            continue

        subdf['sample_id'] = subdf['sample_id'].astype('category')
        subdf = subdf[~pd.isna(subdf["score"])]

        print("FItting for model_id: {}".format(model_id))
        lmm = smf.mixedlm("score ~ condition", subdf, groups=subdf["sample_id"])
        res = lmm.fit()

        # lm = smf.ols("score ~ condition", data=subdf).fit()
        output[model_id] = {
            'coef_effective': res.params.get("condition[T.effective]", np.nan),
            'pval': res.pvalues.get("condition[T.effective]", np.nan),
            'conf_int': res.conf_int().loc["condition[T.effective]"].tolist(),
            "n": len(subdf)
        }

    print(f"Results for {rde_type} RDE:")
    for model_id, res in output.items():
        print(
            f"Model {model_id}: Coef: {res['coef_effective']}, p-value: {res['pval']}, CI: {res['conf_int']} ({res['n']} samples)")

    pvals = [res['pval'] for res in output.values()]
    rej, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    # Attach corrected p-values
    for i, key in enumerate(output.keys()):
        output[key]['pval_corrected'] = pvals_corr[i]
        output[key]['significant'] = rej[i]

    print(f"Corrected p-values for {rde_type} RDE:")
    for model_id, res in output.items():
        print(f"Model {model_id}: Corrected p-value: {res['pval_corrected']}, Significant: {res['significant']}")

    return output


def analysis(models, delta_evals):
    soundness_neutral_data = {}
    soundness_effective_data = {}
    for delta_type, stats in delta_evals.items():
        if delta_type not in soundness_neutral_data:
            soundness_neutral_data[delta_type] = {}

        if delta_type not in soundness_effective_data:
            soundness_effective_data[delta_type] = {}

        for cf_type, cfstats in stats.items():
            cf_is_soundness_neutral = cf_type in SOUNDNESS_NEUTRAL
            tba = soundness_neutral_data if cf_is_soundness_neutral else soundness_effective_data

            data = get_data_per_argtor(cfstats)
            for argtor, argdata in data.items():
                tba[delta_type][argtor] = tba[delta_type].get(argtor, []) + argdata

    print("---" * 20 + "PointRDE" + "---" * 20)
    res_point = mixed_linear_effects(models, "PointRDE", soundness_neutral_data, soundness_effective_data)

    print("---" * 20 + "AspectRDE" + "---" * 20)
    res_aspect = mixed_linear_effects(models, "AspectRDE", soundness_neutral_data, soundness_effective_data)

    print("---" * 20 + "ScoreRDE" + "---" * 20)
    res_score = mixed_linear_effects(models, "ScoreRDE", soundness_neutral_data, soundness_effective_data)

    print("---" * 20 + "Latex Table" + "---" * 20)
    print("Latex table")

    for model_id, resp in res_point.items():
        resa = res_aspect.get(model_id)
        ress = res_score.get(model_id)

        print(
            f"{model_id} & ${resa['pval']:.3f}$ & ${resa['pval_corrected']:.3f}$ & ${resp['pval']:.3f}$ & ${resp['pval_corrected']:.3f}$ & ${ress['pval']:.3f}$ & ${ress['pval_corrected']:.3f}$ \\\\")


def main():
    parser = argparse.ArgumentParser(description="Compute ATE")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument("--models", type=str, nargs="+", required=False, help="List of models to consider.")

    args = parser.parse_args()

    if args.models is None:
        models = [
            "ablator_systematic_guideline-gpt-4o-mini",
            "systematic_guideline-gpt-4o-mini",
            "systematic_guideline-ollama_deepseek-r1:14b",
            "systematic_guideline-ollama_phi4:latest",
            "systematic_generic-gpt-4o-mini",
            "systematic_generic-gpt-4.1",
            "systematic_generic-ollama_deepseek-r1:14b",
            "systematic_guideline-deepseekv3",
            "systematic_generic-deepseekv3",

            "systematic_generic-ollama_phi4:latest",
            "treereviewer-gpt-4o-mini",
            "reviewer2-gpt-4o-mini",
            "deepreviewer-gpt-4o-mini"
        ]
    else:
        models = args.models

    delta_evals = load_delta_evals(args.data_path + "/dev")
    delta_evals_t = load_delta_evals(args.data_path + "/test")

    delta_evals = combine_delta_subsets(delta_evals, delta_evals_t)
    analysis(models, delta_evals)

