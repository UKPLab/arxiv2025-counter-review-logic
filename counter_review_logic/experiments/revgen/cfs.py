import argparse
import logging
import random
import sys
import time
from pathlib import Path

import yaml

from cerg.experiments.utils import get_train_data, load_llm, load_argtor, set_all_seeds, get_data_by_share
from cerg.framework.argtor import AutomaticReviewGenerator, AutomaticReviewDataset
from cerg.pipeline import generate_reviews_for_originals, generate_reviews_for_counterfactuals


def experiment(
        dataset_path: str | Path,
        venues: list[str],
        argtor: AutomaticReviewGenerator,
        output_path: str | Path,
        split: str):
    papers = get_data_by_share(dataset_path, split, merge=False)

    # load all venues if not otherwise requested
    if venues is None:
        venues = list(papers.keys())

    assert all([v in papers for v in
                venues]), f"Not all venues are in the dataset. Provided: {venues}, Available: {list(papers.keys())}"

    # get papers and shuffle them
    papers = [p for k, v in papers.items() for p in v if k in venues]
    random.shuffle(papers)

    # run pipeline with only original papers
    out_path = Path(output_path) / split / "reviews"
    out_path.mkdir(parents=True, exist_ok=True)

    # load cf_dataset_paths
    cf_dataset_path = Path(output_path) / split / "cf_datasets"
    cf_dataset_paths = [p for p in cf_dataset_path.iterdir() if p.is_dir()]

    # generate reviews
    review_paths = []
    for dp in cf_dataset_paths:
        logging.info(f"Generating reviews for dataset split: {dp} with {len(papers)} papers.")

        review_paths += [
            generate_reviews_for_counterfactuals(papers, dp, argtor, out_path, True)]

    return review_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pilot experiment with specified parameters.")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--argtor_type', type=str, required=True, help='Type of argtor to use.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save the results.')
    parser.add_argument('--split', type=str, choices=["train", "dev", "test"], required=True,
                        help='The split of the data')
    parser.add_argument('--seed', type=int, required=False, default=10203, help='Seed for expreiments.')

    parser.add_argument('--temperature', type=float, required=False, help='Temperature for sampling from the LLM.')
    parser.add_argument('--config', type=str, required=False, help='General config for the LLM')
    parser.add_argument('--model_type', type=str, required=False,
                        help='Type of the model to use. Provide if applicable')
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

    # load llm if applicable
    config = {}
    if args.argtor_type not in ["marg"]:
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
    argtor = load_argtor(name=args.argtor_type + f"-{args.model_type}" if args.model_type else "",
                         argtor_type=args.argtor_type,
                         config=config)

    # run experiment
    logging.info("Running automatic review generation.")
    started = time.time()
    res, res_path = experiment(args.data_path,
                               args.venues,
                               argtor,
                               args.results_dir,
                               args.split)
    ended = time.time()

    logging.info("Finished experiment. Results saved at: " + str(res_path))
    logging.info("Generated {} reviews.".format(len(res)))
