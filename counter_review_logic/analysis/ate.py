import argparse
from copy import deepcopy

import pandas as pd

from .utils import SOUNDNESS_NEUTRAL, get_data_per_argtor, DELTA_KEYS, stats_on_merged_data, load_delta_evals, \
    combine_delta_subsets


def ate(soundness_effective_data, soundness_neutral_data, output_path):
    soundness_effective_data = deepcopy(soundness_effective_data)

    result = {}
    print("=== Analysis")
    print("********** Soundness Neutral Deltas")
    for delta_type, data in soundness_neutral_data.items():
        if delta_type not in DELTA_KEYS:
            continue

        stats = stats_on_merged_data(data, DELTA_KEYS[delta_type])

        print(">>>>>> DELTA", delta_type)
        for argtor, v in stats.items():
            print("#### ARGTOR", argtor)
            pd.DataFrame.from_records(data[argtor]).to_csv(f"{output_path}/neutral_{delta_type}_{argtor}.csv")

            if argtor not in result:
                result[argtor] = {}

            if delta_type not in result[argtor]:
                result[argtor][delta_type] = {}

            result[argtor][delta_type]["neutral"] = f"${v['mean']:.2f}" + "\pm" + f"{v['std']:.2f}$"
            result[argtor][delta_type]["neutral_num"] = v['mean']

    print("********** Soundness Effective Deltas")
    for delta_type, data in soundness_effective_data.items():
        if delta_type not in DELTA_KEYS:
            continue
        stats = stats_on_merged_data(data, DELTA_KEYS[delta_type])

        print(">>>>>> DELTA", delta_type)
        for argtor, v in stats.items():
            print("#### ARGTOR", argtor)
            pd.DataFrame.from_records(data[argtor]).to_csv(f"{output_path}/effective_{delta_type}_{argtor}.csv")

            if argtor not in result:
                result[argtor] = {}

            if delta_type not in result[argtor]:
                result[argtor][delta_type] = {}

            result[argtor][delta_type]["soundndess_effective"] = f"${v['mean']:.2f}" + "\pm" + f"{v['std']:.2f}$"
            result[argtor][delta_type]["difference"] = v["mean"] - result[argtor][delta_type]["neutral_num"]
            del result[argtor][delta_type]["neutral_num"]


    result_recs = []
    for argtor, d1 in result.items():
        rec = {
            "argtor": argtor
        }
        for delta_type, d2 in d1.items():
            for group, value in d2.items():
                rec[f"{delta_type}_{group}"] = value
        result_recs += [rec]

    pd.DataFrame.from_records(result_recs).to_csv(f"{output_path}/ate_overview.csv")


def z_ranking(soundness_effective_data, soundness_neutral_data, output_path):
    soundness_effective_data = deepcopy(soundness_effective_data)

    result = {}
    print("=== Analysis")
    print("********** Soundness Neutral Deltas")
    for delta_type, data in soundness_neutral_data.items():
        if delta_type not in DELTA_KEYS:
            continue

        stats = stats_on_merged_data(data, DELTA_KEYS[delta_type])

        print(">>>>>> DELTA", delta_type)
        for argtor, v in stats.items():
            print("#### ARGTOR", argtor)
            if argtor not in result:
                result[argtor] = {}

            if delta_type not in result[argtor]:
                result[argtor][delta_type] = {}

            result[argtor][delta_type]["neutral_mean"] = v["mean"]
            result[argtor][delta_type]["neutral_std"] = v['std']

    print("********** Soundness Effective Deltas")
    for delta_type, data in soundness_effective_data.items():
        if delta_type not in DELTA_KEYS:
            continue
        stats = stats_on_merged_data(data, DELTA_KEYS[delta_type])

        print(">>>>>> DELTA", delta_type)
        for argtor, v in stats.items():
            print("#### ARGTOR", argtor)
            pd.DataFrame.from_records(data[argtor]).to_csv(f"{output_path}/effective_{delta_type}_{argtor}.csv")

            if argtor not in result:
                result[argtor] = {}

            if delta_type not in result[argtor]:
                result[argtor][delta_type] = {}

            result[argtor][delta_type]["soundness_mean"] = v['mean']

            std = result[argtor][delta_type]["neutral_std"]
            std = std if std > 0 else 1.0
            result[argtor][delta_type]["rank_score"] = abs(result[argtor][delta_type]["soundness_mean"] - result[argtor][delta_type]["neutral_mean"]) / std

    result_recs = []
    for argtor, d1 in result.items():
        rec = {
            "argtor": argtor
        }
        for delta_type, d2 in d1.items():
            rec[f"{delta_type}"] = d2["rank_score"]
        result_recs += [rec]

    pd.DataFrame.from_records(result_recs).to_csv(f"{output_path}/z_ranking.csv")


def analysis(delta_evals, output_path):
    # gather data
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

    ate(soundness_effective_data, soundness_neutral_data, output_path)
    z_ranking(soundness_effective_data, soundness_neutral_data, output_path)


def main():
    parser = argparse.ArgumentParser(description="Compute ATE")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save the results.')

    args = parser.parse_args()

    delta_evals = load_delta_evals(args.data_path + "/dev")
    delta_evals_t = load_delta_evals(args.data_path + "/test")

    delta_evals = combine_delta_subsets(delta_evals, delta_evals_t)
    analysis(delta_evals, args.results_dir)


if __name__ == "__main__":
    main()