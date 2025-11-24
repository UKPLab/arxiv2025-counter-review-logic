import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

from counter_review_logic.experiments.utils import set_all_seeds, load_llm, load_cfgen, load_rcd, load_rde, \
    run_experiment_stage


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Counter Review Logic")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["blueprints", "counterfactuals", "evaluation", "analyze"],
        help="The experiment to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the configuration file stored as a JAML.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input data."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output data."
    )

    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
            print("Config loaded successfully:", config)
    except Exception as e:
        print(f"Error loading config file: {e}")
        config = None

    if "llm_api_endpoint" in config:
        os.environ["ENDPOINT"] = config["llm_api_endpoint"]
    if "llm_api_key" in config:
        os.environ["API_KEY"] = config["llm_api_key"]
    if "llm_api_version" in config:
        os.environ["API_VERSION"] = config["llm_api_version"]
    if "llm_cost_cache_dir" in config:
        os.environ["COST_CACHE_DIR"] = config["llm_cost_cache_dir"]
    os.environ["PROMPT_DIR"] = str((Path(__file__).resolve().parent.parent / "prompts").absolute())

    # set seeds
    seed = config.get("seed", 10203)
    set_all_seeds(seed)

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Log to stdout
        ]
    )

    logging.info("SEED set to {}".format(seed))

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    def create_llm():
        # llm loading
        if "seed" in config and "llm_config" in config:
            config["llm_config"]["seed"] = config["seed"]

        llm = load_llm(name=config.get("llm_type"),
                       llm_type=config.get("llm_type"),
                       config=config.get("llm_config", None))
        return llm

    if args.experiment == "blueprints":
        from counter_review_logic.experiments.create_blueprints import experiment

        assert config is not None, "Config file needed for blueprint creation"

        # llm loading
        if "seed" in config and "llm_config" in config:
            config["llm_config"]["seed"] = config["seed"]

        llm = create_llm()

        experiment(
            dataset_path=args.input_path + "/papers",
            venues=config.get("venues", None),
            llm=llm,
            output_path=args.output_path,
        )
        logging.info(f"DONE! Blueprints written to {args.output_path}/blueprint")
    elif args.experiment == "counterfactuals":
        from counter_review_logic.experiments.create_cfs import experiment

        assert config is not None, "Config file needed for counterfactual creation"

        llm = create_llm()

        cfgen = load_cfgen(cfgen_type=config.get("cf_type"), config={"llm": llm, **config.get("cfgen_config", {})})

        if "blueprint_path" in config:
            os.environ["BLUEPRINT_DIR"] = config["blueprint_path"]

        experiment(
            dataset_path=args.input_path + "/papers",
            venues=config.get("venues", None),
            cfgen=cfgen,
            output_path=args.output_path,
            forced=True  # forcing override as default
        )
    elif args.experiment == "evaluate":
        from counter_review_logic.experiments.detect_deltas import experiment as detect_deltas
        from counter_review_logic.experiments.estimate_effect import experiment as estimate_effects

        assert config is not None, "Config file needed for evaluation"

        # load review change detector
        rcd = load_rcd(rcd_type=config.get("review_difference_dimension"), config={"llm": config["llm"]})

        # detect review changes
        deltas_path = detect_deltas(
            dataset_path=args.input_path + "/papers",
            venues=config.get("venues", None),
            rcd=rcd,
            output_path=args.results_dir)

        logging.info("Review differences detected and written to {}".format(deltas_path))

        # estimate effects
        rde = load_rde(rde_type=config.get("review_difference_dimension"), config={"llm": config["llm"]})

        # estimate effects
        effects_path = estimate_effects(
            dataset_path=args.input_path + "/papers",
            venues=config.get("venues", None),
            rde=rde,
            rcd_name=rcd.name,
            output_path=args.output_path
        )

        logging.info("Review effects estimated and written to {}".format(effects_path))
    elif args.experiment == "analyze":
        run_experiment_stage(
            stage_name="ate_zrank",
            input_path=args.input_path,
            output_path=Path(args.output_path) / "results"
        )