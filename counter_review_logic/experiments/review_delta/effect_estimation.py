import argparse
import logging
import random
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from cerg.experiments.utils import load_llm, load_argtor, set_all_seeds, get_data_by_share, load_rcd, load_rde
from cerg.framework.eval import ReviewDeltaEvaluator
from cerg.framework.rcd import ReviewChangeDetector
from cerg.pipeline import determine_review_deltas, evaluate_review_deltas


def experiment(
        dataset_path: str | Path,
        venues: list[str],
        rde: ReviewDeltaEvaluator,
        rcd_name: str | list[str],
        output_path: str | Path,
        split: str,
        argtors: list[str] = None):
    papers = get_data_by_share(dataset_path, split, merge=False)

    # load all venues if not otherwise requested
    if venues is None:
        venues = list(papers.keys())

    assert all([v in papers for v in
                venues]), f"Not all venues are in the dataset. Provided: {venues}, Available: {list(papers.keys())}"

    # get papers and shuffle them
    papers = [p for k, v in papers.items() for p in v if k in venues]
    random.shuffle(papers)

    # load cf_dataset_paths
    cf_dataset_path = Path(output_path) / split / "cf_datasets"
    cf_dataset_paths = [p for p in cf_dataset_path.iterdir() if p.is_dir()]

    # load review paths
    review_path = Path(output_path) / split / "reviews"
    review_paths = [p for p in review_path.iterdir() if p.is_dir() if p.name != "original"]
    original_review_path = review_path / "original"

    # load argtors
    argtors_found = []
    for p in review_paths:
        for k in p.iterdir():
            if k.is_dir():
                argtors_found += [k.name]

    argtors_found = list(set(a for a in argtors_found if argtors is None or a in argtors))

    if len(argtors_found) == 0:
        raise ValueError("No argtors found in the review paths. Please check the review generation step.")

    delta_dir = Path(output_path) / split / "deltas"
    delta_paths = []
    for p in delta_dir.iterdir():
        if not p.is_dir():
            continue

        if type(rcd_name) == str:
            can = [f for f in p.iterdir() if f.is_file() and f.suffix == ".json" and rcd_name in f.name.lower()]

            if len(can) != 1:
                raise ValueError(
                    "Trying to load delta files, but found more than one or no file in the delta directory matching the delta name: " + rde.name + ". Please check the delta directory. Found: " + str(
                        can) + ".")

            delta_paths += [can[0]]
        else:  # list
            ds = {}
            for rcdn in rcd_name:
                can = [f for f in p.iterdir() if f.is_file() and f.suffix == ".json" and rcdn in f.name.lower()]

                if len(can) != 1:
                    raise ValueError(
                        "Trying to load delta files, but found more than one or no file in the delta directory matching the delta name: " + rde.name + ". Please check the delta directory. Found: " + str(
                            can) + ".")
                ds[rcdn] = can[0]

            delta_paths += [ds]

    print("LOADING DELTAS", delta_paths)

    eval_dir = Path(output_path) / split / "eval"
    eval_dir.mkdir(parents=False, exist_ok=True)

    assert len(review_paths) == len(cf_dataset_paths) == len(
        delta_paths), "Mismatch in number of review paths, counterfactual dataset paths, and delta paths. "

    eval_paths = []
    for reviews_path, cf_path, delta_path in zip(review_paths, cf_dataset_paths, delta_paths):
        eval_paths += [evaluate_review_deltas(papers,
                                              delta_path,
                                              cf_path,
                                              argtors_found,
                                              rde.name,
                                              rde,
                                              eval_dir,
                                              True)]

    return eval_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pilot experiment with specified parameters.")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--review_delta_type', type=str, required=True,
                        help='Type of review deltas')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save the results.')
    parser.add_argument('--split', type=str, choices=["train", "dev", "test"], required=True,
                        help='The split of the data')
    parser.add_argument('--seed', type=int, required=False, default=10203, help='Seed for expreiments.')

    parser.add_argument('--config', type=str, required=False, help='General config for the LLM')
    parser.add_argument('--venues', type=str, nargs='+', required=False, help='List of venues.')

    args = parser.parse_args()

    # set seed
    set_all_seeds(args.seed)

    # set log level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Log to stdout
        ]
    )

    logging.info("SEED set to {}".format(args.seed))
    logging.info("Input path {}".format(args.data_path))
    logging.info("Target path {}".format(args.results_dir))

    # load review change detector
    rde = load_rde(rde_type=args.review_delta_type)

    if args.review_delta_type == "soundness":
        rcds = ["point", "aspect"]
    else:
        rcds = args.review_delta_type

    # run experiment
    logging.info("Running delta evaluation.")
    started = time.time()
    res_path = experiment(args.data_path,
                          args.venues,
                          rde,
                          rcds,
                          args.results_dir,
                          args.split)
    ended = time.time()

    logging.info("Finished experiment. Results saved at: " + str(res_path))
