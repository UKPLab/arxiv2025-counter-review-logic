import argparse
import logging
import random
import sys
import time
from pathlib import Path

import yaml

from .utils import load_llm, set_all_seeds, get_data_by_share, load_cfgen
from ..framework import PaperCounterfactualGenerator, PaperCounterfactualDataset
from .pipeline import create_cf_dataset


def experiment(
        dataset_path: str | Path,
        venues: list[str],
        cfgen: PaperCounterfactualGenerator,
        output_path: str | Path,
        forced: bool):
    papers = get_data_by_share(dataset_path, "test", merge=False)

    # load all venues if not otherwise requested
    if venues is None:
        venues = list(papers.keys())

    assert all([v in papers for v in
                venues]), f"Not all venues are in the dataset. Provided: {venues}, Available: {list(papers.keys())}"

    # get papers and shuffle them
    papers = [p for k, v in papers.items() for p in v if k in venues]
    random.shuffle(papers)

    # run pipeline with only original papers
    out_path = Path(output_path) / "cf_datasets"
    out_path.mkdir(parents=True, exist_ok=True)

    # generate cfs
    result_path = create_cf_dataset(
        paper_dataset=papers,
        cf_generators=cfgen,
        out_path=out_path,
        cached=not forced
    )

    # load and return
    cfs = PaperCounterfactualDataset.load(fp=result_path[0], originals=papers)

    return cfs, result_path


def main():
    parser = argparse.ArgumentParser(description="Create counterfactual dataset with the specified type and model.")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--blueprint_path', type=str, required=True, help='Path to the blueprint directory.')
    parser.add_argument('--cftype', type=str, required=True, help='Type of CF to use.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save the results.')
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

    logging.info("Loading LLM.")

    if args.config:
        with open(args.config, 'r') as file:
            llm_config = yaml.safe_load(file)
    else:
        llm_config = None

    config["llm"] = load_llm(name="",
                             llm_type=args.model_type,
                             config=llm_config)

    logging.info(f"Loaded LLM {config['llm'].name}.")

    if args.temperature:
        logging.info("Setting temperature for LLM to {}".format(args.temperature))
        config["llm"].add_config({"temperature": args.temperature})

    # laod argtor
    if args.cftype.startswith("blueprint_"):
        config["name"] = args.cftype + "_" + config["llm"].name

    cfgen = load_cfgen(cfgen_type=args.cftype, config=config)

    # run experiment
    logging.info("Running automatic review generation.")
    started = time.time()
    res, res_path = experiment(args.data_path,
                               args.venues,
                               cfgen,
                               args.results_dir,
                               args.force if args.force else False)
    ended = time.time()

    logging.info("Finished experiment. Results saved at: " + str(res_path))
    logging.info("Generated {} reviews.".format(len(res)))


if __name__ == "__main__":
    main()