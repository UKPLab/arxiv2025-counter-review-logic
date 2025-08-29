import argparse
import logging
import random
import sys
import time
from pathlib import Path

import yaml

from .utils import load_llm, load_argtor, set_all_seeds, get_data_by_share
from ..framework import AutomaticReviewGenerator, AutomaticReviewDataset
from .pipeline import generate_reviews_for_originals


def experiment(
        dataset_path: str | Path,
        venues: list[str],
        argtor: AutomaticReviewGenerator,
        output_path: str | Path,
        split: str,
        forced: bool):
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

    # generate reviews
    result_path = generate_reviews_for_originals(papers, argtor, out_path, not forced, {})

    # load and return reviews
    reviews = AutomaticReviewDataset.load(result_path)

    return reviews, result_path


def main():
    parser = argparse.ArgumentParser(description="Run pilot experiment with specified parameters.")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--argtor_type', type=str, required=True, help='Type of argtor to use.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save the results.')
    parser.add_argument('--split', type=str, choices=["train", "dev", "test"], required=True,
                        help='The split of the data')
    parser.add_argument('--seed', type=int, required=False, default=10203, help='Seed for expreiments.')
    parser.add_argument('--force', type=bool, required=False, default=False, help='Force overrride of data.')

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

    # load llm if applicable
    config = {}
    if args.argtor_type not in ["marg"]:
        logging.info("Loading LLM.")

        if args.config:
            with open(args.config, 'r') as file:
                llm_config = yaml.safe_load(file)
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
                     args.split,
                     args.force if args.force else False)
    ended = time.time()

    logging.info("Finished experiment. Results saved at: " + str(res_path))
    logging.info("Generated {} reviews.".format(len(res)))


if __name__ == "__main__":
    main()