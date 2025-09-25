import argparse
import json
import logging
import sys
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from . import generate_reviews_for_originals, generate_reviews_for_counterfactuals, determine_review_deltas
from .pipeline import evaluate_review_deltas
from ..analysis import load_delta_evals
from ..analysis.ate import analysis as ate_analysis
from ..data import load_paper_datasets
from ..llm import get_num_ctx_of_llm
from ..model.cfgen import BlueprintBasedCF


def get_data_by_share(dataset_dir: str | Path, share: str, merge: bool):
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)

    # load splits
    with open(dataset_dir / "split.json", "r") as f:
        splits = json.load(f)

    split_pids = splits[share]

    # load papers
    papers = load_paper_datasets(dataset_dir, venues=None, by_split=False, pids=split_pids)

    if merge:
        return [p for v in papers for p in papers[v]]
    else:
        return papers


def get_dev_data(dataset_dir, merge=False):
    """Get the dev data from the dataset directory. If merge is True, the data from all venues will be merged."""

    return get_data_by_share(dataset_dir, "dev", merge)


def get_test_data(dataset_dir, merge=False):
    """Get the test data from the dataset directory. If merge is True, the data from all venues will be merged."""

    return get_data_by_share(dataset_dir, "test", merge)


def load_cfgen(cfgen_type, config):
    assert "llm" in config, "LLM is required for CFGen"

    if cfgen_type == "active2passive":
        from ..model.cfgen.ActivePassiveCF import ActivePassiveCF
        return ActivePassiveCF(**config)
    elif cfgen_type == "british2american":
        from ..model.cfgen.BritishAmericanCF import BritishAmericanCF
        return BritishAmericanCF(**config)
    elif cfgen_type == "language_error":
        from ..model.cfgen.LanguageErrorCF import LanguageErrorCF
        return LanguageErrorCF(**config)
    elif cfgen_type == "layout":
        from ..model.cfgen.PaperLayoutCF import PaperLayoutCF
        if "llm" in config:
            del config["llm"]

        return PaperLayoutCF(**config)
    elif cfgen_type.startswith("blueprint_"):
        assert "name" in config, "Blueprint CFGen requires a name in the config"

        p = None
        if "result" in cfgen_type:
            from ..model.blueprint.ResultPerturbator import ResultPerturbator
            p = ResultPerturbator(**config)
        elif "conclusion" in cfgen_type:
            from ..model.blueprint.ConclusionPerturbator import ConclusionPerturbator
            p = ConclusionPerturbator(**config)
        elif "finding" in cfgen_type:
            from ..model.blueprint.FindingPerturbator import FindingPerturbator
            p = FindingPerturbator(**config)

        cfgen = BlueprintBasedCF(name=cfgen_type)
        cfgen.set_perturbator(p)

        return cfgen
    else:
        raise ValueError(f"Unknown CFGen type: {cfgen_type}")


def load_argtor(name, argtor_type, config=None):
    """
    Load the specified ARGtor.

    :param name:
    :param argtor_type:
    :return:
    """
    if config is None:
        config = {}

    print("LOADING ARGTOR", argtor_type, type(argtor_type), config)

    if argtor_type == "liangetal2024":
        from ..model.argtor import Liangetal2024ARGtor
        return Liangetal2024ARGtor(name=name, **config)
    elif argtor_type == "marg":
        from ..model.argtor import MARG
        return MARG(name=name, **config)
    elif argtor_type == "review_critique":
        from ..model.argtor import ReviewCritiqueARGtor
        return ReviewCritiqueARGtor(name=name, **config)
    elif argtor_type == "shinetal2025":
        from ..model.argtor import Shinetal2025ARGtor
        return Shinetal2025ARGtor(name=name, **config)
    elif argtor_type == "deepreviewer":
        from ..model.argtor import DeepReviewerARGtor
        return DeepReviewerARGtor(name=name, **config)
    elif argtor_type == "reviewer2":
        from ..model.argtor import Reviewer2ARGtor
        return Reviewer2ARGtor(name=name, **config)
    elif argtor_type == "treereviewer":
        from ..model.argtor import TreeReviewARGtor
        return TreeReviewARGtor(name=name, **config)
    elif "systematic" in argtor_type:
        sub_type = argtor_type.split("_")[-1]
        from ..model.argtor import SystematicARGtor
        return SystematicARGtor(name=name, prompt_type=sub_type, **config)
    else:
        raise ValueError(f"Unknown ARGtor type: {argtor_type}")


def load_rcd(rcd_type, config=None):
    """
    Load the specified RCD.

    :param name:
    :param argtor_type:
    :return:
    """
    if config is None:
        config = {}

    if rcd_type == "polarity":
        from ..model.rcd.PolarityRCD import PolarityRCD
        return PolarityRCD(**config)
    elif rcd_type == "score":
        from ..model.rcd.ScoreRCD import ScoreRCD
        return ScoreRCD()
    elif rcd_type == "surface":
        from ..model.rcd.SurfaceRCD import SurfaceRCD
        return SurfaceRCD()
    elif rcd_type == "aspect":
        from ..model.rcd.AspectRCD import AspectRCD
        return AspectRCD()
    else:
        raise ValueError(f"Unknown ARGtor type: {rcd_type}")


def load_rde(rde_type, config=None):
    """
    Load the specified RDE

    :param name:
    :param argtor_type:
    :return:
    """
    if config is None:
        config = {}

    if rde_type == "score":
        from ..model.rde.ScoreRDE import ScoreRDE
        return ScoreRDE()
    elif rde_type == "surface":
        from ..model.rde.SurfaceRDE import SurfaceRDE
        return SurfaceRDE()
    elif rde_type == "aspect":
        from ..model.rde.AspectRDE import AspectRDE
        return AspectRDE()
    elif rde_type in ["polarity", "point"]:
        from ..model.rde.PointRDE import PointRDE
        return PointRDE()
    elif rde_type == "soundness":
        from ..model.rde.SoundnessRDE import SoundnessRDE
        return SoundnessRDE()
    else:
        raise ValueError(f"Unknown ARGtor type: {rde_type}")


def load_llm(name, llm_type, cost_path=None, config=None):
    """
    Load the specified LLM.

    :param name:
    :param llm_type:
    :param cost_path:
    :return:
    """
    def_path = Path("/storage/ukp/shared/shared_model_weights/")

    if llm_type.startswith("gpt"):  # openai
        from ..llm.OpenAi import OpenAiChatLLM
        return OpenAiChatLLM(name=f"{name}_{llm_type}",
                             model=llm_type,
                             cost_cache_dir=cost_path,
                             config=config)
    elif llm_type == "deepseekv3":
        from ..llm.DeepSeek import DeepSeekChatLLM
        return DeepSeekChatLLM(name=f"{name}_{llm_type}",
                               model=llm_type,
                               cost_cache_dir=cost_path,
                               config=config)
    elif llm_type.lower().startswith("ollama_"):
        llm_type = llm_type[len("ollama_"):]

        from ..llm.OllamaChatLLM import OllamaChatLLM

        if config is None:
            config = {}

        config["num_ctx"] = get_num_ctx_of_llm(llm_type)

        return OllamaChatLLM(name=f"{name.replace('/', '.')}_{llm_type}",
                             config=config,
                             model=llm_type)
    elif "mixtral" in llm_type.lower():
        raise NotImplementedError("mixtral models cannot be loaded yet")
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def set_all_seeds(seed: int):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU


def default_experiment_argparse(experiment_name: str):
    parser = argparse.ArgumentParser(description=f"Run {experiment_name} with specified parameters.")

    # basics
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save the results.')
    parser.add_argument('--split', type=str, choices=["train", "dev", "test"], required=True,
                        help='The split of the data')

    # general
    parser.add_argument('--seed', type=int, required=False, default=10203, help='Seed for expreiments.')
    parser.add_argument('--force', type=bool, required=False, default=False, help='Force overrride of data.')

    # llm
    parser.add_argument('--llm', type=str, required=False, help='Type of the model to use. Provide if applicable')
    parser.add_argument('--llm_config', type=str, required=False, default=False, help='Path to the llm config file.')
    parser.add_argument('--temperature', type=float, required=False, help='Temperature for sampling from the LLM.')

    # subsampling
    parser.add_argument('--venues', type=str, nargs='+', required=False, help='List of venues.')

    return parser


def default_experiment_setup(args):
    result = {}

    # set log level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Log to stdout
        ]
    )

    # set seed
    if args.seed:
        set_all_seeds(args.seed)
        logging.info("Setting seed to {}".format(args.seed))
    else:
        logging.info("No seed provided. Not fixing seed at all. Discouraged!")

    # load config
    if args.llm and args.llm_config:
        logging.info("Loading LLM config from {}".format(args.llm_config))

        with open(args.llm_config, 'r') as file:
            llm_config = yaml.safe_load(file)

        if args.seed:
            logging.info("Overriding seed for LLM to {}".format(args.seed))
            llm_config["seed"] = args.seed

        if args.temperature:
            logging.info("Overriding temperature for LLM to {}".format(args.temperature))
            llm_config["temperature"] = args.temperature
    else:
        logging.info("No LLM config provided. Using default and/or server-side config.")
        llm_config = {}

    # load LLM
    if args.llm:
        logging.info("Loading LLM {}.".format(args.llm))

        result["llm"] = load_llm(name=args.llm, llm_type=args.llm, config=llm_config)

    return result


def run_experiment_stage(stage_name, input_path:Path, output_path:Path, **kwargs):
    # Load the original papers
    paper_dataset_path = input_path / "papers"

    papers = get_test_data(paper_dataset_path, merge=True)
    papers = [p for k, v in papers.items() for p in v]
    random.shuffle(papers)

    if stage_name == "generate_reviews":
        assert "argtor" in kwargs, "argtor is required for generating reviews"

        # create output dir if necessary
        reviews_path = output_path / "reviews"
        reviews_path.mkdir(parents=True, exist_ok=True)

        # Run your arg on the original papers
        generate_reviews_for_originals(papers,
                                       kwargs["argtor"],
                                       reviews_path,
                                       cached=False,
                                       argtor_config=kwargs["argtor_config"])  # pass any config you need

        # Load counterfactual dataset paths
        cf_dataset_paths = [p for p in (input_path / "cf_dataset").iterdir() if p.is_dir()]

        # Run your arg on the counterfactual papers
        for dp in cf_dataset_paths:
            generate_reviews_for_counterfactuals(papers,
                                                 dp,
                                                 kwargs["argtor"],
                                                 reviews_path,
                                                 cached=False,
                                                 argtor_config=kwargs["argtor_config"] if "argtor_config" in kwargs else {})  # pass config
    elif stage_name == "diff_reviews":
        assert "argtor" in kwargs, "argtor name is required for evaluating reviews"
        assert "rcd" in kwargs, "rcd is required for evaluating reviews"

        original_reviews_path = output_path / "reviews" / "original"
        cf_reviews_path = [p for p in (output_path / "reviews").iterdir() if p.is_dir() and p.name != "original"]
        cf_dataset_paths = [(input_path / "cf_datasets" / p.name) for p in cf_reviews_path]

        # create output path if necessary
        delta_dir = output_path / "deltas"
        delta_dir.mkdir(parents=False, exist_ok=True)

        delta_paths = []
        for rp, dp in zip(cf_reviews_path, cf_dataset_paths):
            delta_paths += [
                determine_review_deltas(papers,
                                        rp,
                                        original_reviews_path,
                                        kwargs["argtor"],
                                        dp,
                                        kwargs["rcd"],
                                        delta_dir,
                                        cached=False,
                                        config=kwargs["rcd_config"] if "rcd_config" in kwargs else {})]

    elif stage_name == "compare_reviews":
        assert "argtor" in kwargs, "ARGtor name expected as an argument 'argtor'"
        assert "rcd" in kwargs, "RCD name is expected as an argument 'rcd'"
        assert "rde" in kwargs, "RDE is expected as an argument 'rde'"

        cf_reviews_path = [p for p in (output_path / "reviews").iterdir() if p.is_dir() and p.name != "original"]
        cf_dataset_paths = [(input_path / "cf_datasets" / p.name) for p in cf_reviews_path]

        # get deltas
        delta_paths = []
        for cfp in cf_dataset_paths:
            cfn = cfp.name
            delta_paths += [output_path / "deltas" / cfn / (kwargs["rcd"].name + ".json")]

        for dp in delta_paths:
            assert dp.exists() and dp.isfile(), f"{dp} does not exist as a delta type, failed to load it."

        # create output path if necessary
        eval_dir = output_path / "eval"
        eval_dir.mkdir(parents=False, exist_ok=True)

        eval_paths = []
        for reviews_path, cf_path, delta_path in zip(cf_reviews_path, cf_dataset_paths, delta_paths):
            eval_paths += [evaluate_review_deltas(papers,
                                                  delta_path,
                                                  cf_path,
                                                  kwargs["argtor"],
                                                  kwargs["rcd"],
                                                  kwargs["rde"],
                                                  eval_dir,
                                                  cached=False,
                                                  config=kwargs["rde_config"] if "rde_config" in kwargs else {})]
    elif stage_name == "ate_zrank":
        evals = load_delta_evals(output_path)
        ate_analysis(evals, output_path / "ate")