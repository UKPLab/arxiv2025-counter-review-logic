import argparse
import asyncio
import logging
import random
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from .utils import load_llm, set_all_seeds, get_data_by_share, load_rcd
from ..framework import ReviewChangeDetector
from .pipeline import determine_review_deltas, adetermine_review_deltas


async def aexperiment(
        dataset_path: str | Path,
        venues: list[str],
        rcd: ReviewChangeDetector,
        output_path: str | Path,
        split: str,
        argtors: list[str] = None):
    """
    Use this function to run the delta detection asynchronously; this helps speed up quite a bit.
    """
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
    delta_dir.mkdir(parents=False, exist_ok=True)

    delta_paths = []
    for rp, dp in tqdm(list(zip(review_paths, cf_dataset_paths)), desc="iterating over cfs"):
        delta_paths += [
            await adetermine_review_deltas(papers, rp, original_review_path, argtors_found, dp, rcd, delta_dir, True)]

    print("DELTAS", delta_paths)

    return delta_paths


def experiment(
        dataset_path: str | Path,
        venues: list[str],
        rcd: ReviewChangeDetector,
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
    delta_dir.mkdir(parents=False, exist_ok=True)

    delta_paths = []
    for rp, dp in tqdm(list(zip(review_paths, cf_dataset_paths)), desc="iterating over cfs"):
        delta_paths += [
            determine_review_deltas(papers, rp, original_review_path, argtors_found, dp, rcd, delta_dir, True)]

    print("DELTAS", delta_paths)

    return delta_paths


def main():
    parser = argparse.ArgumentParser(description="Detect the difference between CF and original reviews per model.")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--argtor_type', type=str, required=False, help='Type of argtor to use.')
    parser.add_argument('--review_delta_type', type=str, required=True, help='Type of argtor to use.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save the results.')
    parser.add_argument('--split', type=str, choices=["train", "dev", "test"], required=True,
                        help='The split of the data')
    parser.add_argument('--seed', type=int, required=False, default=10203, help='Seed for expreiments.')

    parser.add_argument('--temperature', type=float, required=False, help='Temperature for sampling from the LLM.')
    parser.add_argument('--config', type=str, required=False, help='General config for the LLM')
    parser.add_argument('--model_type', type=str, required=False,
                        help='Type of the model to use. Provide if applicable')
    parser.add_argument('--venues', type=str, nargs='+', required=False, help='List of venues.')
    parser.add_argument('--sync', type=bool, required=False, help='List of venues.')

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

    # load llm if applicable
    config = {}
    if args.argtor_type not in ["reviewer2", "marg"]:
        logging.info("Loading LLM.")

        if args.config:
            logging.info("Loading LLM config from: {}".format(args.config))
            with open(args.config, 'r') as file:
                llm_config = yaml.safe_load(file)

            logging.info("LLM config loaded: {}".format(llm_config))
        else:
            llm_config = None

        config["llm"] = load_llm(name=args.model_type,
                                 llm_type=args.model_type,
                                 config=llm_config)

        if args.temperature:
            logging.info("Setting temperature for LLM to {}".format(args.temperature))
            config["llm"].add_config({"temperature": args.temperature})

    # laod argtor
    argtor = args.argtor_type

    # load review change detector
    rcd = load_rcd(rcd_type=args.review_delta_type, config={"llm": config["llm"]})

    # run experiment
    logging.info("Running delta detection.")
    logging.info("Sync mode = {}".format(args.sync))
    started = time.time()
    if args.review_delta_type == "point" and not args.sync:
        res_path = asyncio.run(aexperiment(args.data_path,
                                           args.venues,
                                           rcd,
                                           args.results_dir,
                                           args.split,
                                           argtors=[argtor] if argtor is not None else None))
    else:
        res_path = experiment(args.data_path,
                              args.venues,
                              rcd,
                              args.results_dir,
                              args.split,
                              argtors=[argtor] if argtor is not None else None)

    ended = time.time()

    logging.info("Finished experiment. Results saved at: " + str(res_path))


if __name__ == "__main__":
    main()