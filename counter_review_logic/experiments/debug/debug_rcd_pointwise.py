import argparse
import logging
import random
import sys
from pathlib import Path

from cerg.experiments.utils import get_train_data, load_llm
from cerg.framework.argtor import AutomaticReviewGenerator, AutomaticReviewDataset
from cerg.framework.cfg import PaperCounterfactualGenerator
from cerg.framework.rcd import ReviewChangeDetector
from cerg.models.argtors.Liangetal2024ARGtor import Liangetal2024ARGtor
from cerg.models.cfgens.WhiteSpaces import WhiteSpaceCF
from cerg.models.rcds.AspectRCD import AspectRCD
from cerg.models.rcds.PointOverlapRCD import PointOverlapRCD
from cerg.models.rcds.SurfaceRCD import SurfaceRCD
from cerg.pipeline import full_pipeline


def debug(dataset_path: str | Path,
          venues: list[str],
          cfgen: PaperCounterfactualGenerator,
          argtor: AutomaticReviewGenerator,
          rcd: ReviewChangeDetector,
          output_path: str | Path):
    papers = get_train_data(dataset_path)

    assert all([v in papers for v in
                venues]), f"Not all venues are in the dataset. Provided: {venues}, Available: {list(papers.keys())}"

    papers = [p for k,v in papers.items() for p in v if k in venues]
    random.shuffle(papers)

    # run pipeline with only original papers
    result_paths = full_pipeline(
        paper_dataset=papers,
        paper_dataset_name="+".join(venues),
        cf_generators=[cfgen],
        argtors=[argtor],
        rcd = rcd,
        rde = None,
        experiment_name="debug_rcd",
        out_path=output_path,
    )

    # load and return reviews
    return result_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pilot experiment with specified parameters.")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save the results.')
    parser.add_argument('--venues', type=str, nargs='+', required=False, help='List of venues.')

    args = parser.parse_args()

    # set log level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Log to stdout
        ]
    )

    llm = load_llm("argtor", "gpt-4o-mini")

    # liang et al
    argtor = Liangetal2024ARGtor(name="liangetal2024", llm=llm)

    # cf gen
    cfgen = WhiteSpaceCF()

    # rcd
    #rcd = SurfaceRCD()
    rcd_llm = load_llm("rcd", "ollama_gemma3:27b") # "ollama_deepseek-r1:14b")
    rcd = PointOverlapRCD(rcd_llm)

    # run experiment
    logging.info("Running debug experiment.")
    res = debug(args.data_path,
                args.venues,
                cfgen,
                argtor,
                rcd,
                args.results_dir)

    logging.info("Finished experiment. Results saved at: " + str(res))
