import argparse
import logging
import random
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from ..utils import get_train_data, load_llm, load_argtor, set_all_seeds, get_data_by_share
from cerg.framework.argtor import AutomaticReviewGenerator, AutomaticReviewDataset
from cerg.framework.blueprint import PaperArchitect
from cerg.llms import ChatLLM
from cerg.pipeline import generate_reviews_for_originals


def experiment(
        dataset_path: str | Path,
        venues: list[str],
        llm: ChatLLM,
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
    out_path = Path(output_path) / split / "blueprint" / llm.model
    out_path.mkdir(parents=True, exist_ok=True)

    arch = PaperArchitect(llm, cache_dir=out_path)

    result = []
    start_time= time.time()
    for paper in tqdm(papers, desc="Iterating over papers"):
        result += [arch.construct_blueprint(paper)]
    end_time = time.time()

    logging.info(f"Time taken to construct blueprints: {end_time - start_time} seconds")
    logging.info(f"This means {(end_time - start_time) / len(papers)} seconds per paper")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pilot experiment with specified parameters.")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save the results.')
    parser.add_argument('--split', type=str, choices=["train", "dev", "test"], required=True,
                        help='The split of the data')
    parser.add_argument('--seed', type=int, required=False, default=10203, help='Seed for expreiments.')

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

    if args.config:
        with open(args.config, 'r') as file:
            llm_config = yaml.safe_load(file)
    else:
        llm_config = None

    llm = load_llm(name=args.model_type,
                   llm_type=args.model_type,
                   config=llm_config)


    # run experiment
    logging.info("Running blueprint extraction.")
    started = time.time()
    res = experiment(args.data_path,
                               args.venues,
                               llm,
                               args.results_dir,
                               args.split)
    ended = time.time()

    logging.info("Generated {} reviews.".format(len(res)))
