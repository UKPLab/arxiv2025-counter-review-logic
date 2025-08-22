import argparse
import logging
import random
import sys
from pathlib import Path

from cerg.experiments.utils import get_train_data, load_llm
from cerg.framework.argtor import AutomaticReviewGenerator, AutomaticReviewDataset
from cerg.models.argtors.Liangetal2024ARGtor import Liangetal2024ARGtor
from cerg.pipeline import generate_reviews_for_originals


def debug(dataset_path: str | Path,
          venues: list[str],
          argtor: AutomaticReviewGenerator,
          output_path: str | Path):
    papers = get_train_data(dataset_path)

    assert all([v in papers for v in
                venues]), f"Not all venues are in the dataset. Provided: {venues}, Available: {list(papers.keys())}"

    papers = [p for k,v in papers.items() for p in v if k in venues]
    random.shuffle(papers)

    # run pipeline with only original papers
    out_path = Path(output_path) / f"{argtor.name}" / "reviews"
    out_path.mkdir(parents=True, exist_ok=True)

    # generate reviews
    result_path = generate_reviews_for_originals(papers, argtor, out_path, False, {})

    # load and return reviews
    reviews = AutomaticReviewDataset.load(result_path)

    return reviews


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pilot experiment with specified parameters.")

    parser.add_argument('--cost_path', type=str, required=False, help='Path to the cost cache directory.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--model_type', type=str, required=False, help='Type of the model to use.')
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

    # marg
    # from cerg.models.argtors.marg.MARG import MARG
    # argtor = MARG("debug_marg")

    # create llm
    llm = load_llm("argtor", "gpt-4o-mini")

    # liang et al
    #argtor = Liangetal2024ARGtor(name="debug_liangetal", llm=llm)

    # review critique
    #from cerg.models.argtors.ReviewCritique import ReviewCritiqueARGtor
    #argtor = ReviewCritiqueARGtor(name="debug_review_critique", llm=llm)

    # shin et al
    #from cerg.models.argtors.Shinetal2025ARGtor import Shinetal2025ARGtor
    #argtor = Shinetal2025ARGtor(name="debug_shinetal", llm=llm)

    # deep reviewer
    #from cerg.models.argtors.DeepReviewer import DeepReviewerARGtor
    #argtor = DeepReviewerARGtor(name="debug_deep_reviewer")

    #from cerg.models.argtors.Reviewer2ARGtor import Reviewer2ARGtor
    #argtor = Reviewer2ARGtor(name="debug_reviewer2", llm=llm)

    from cerg.models.argtors.TreeReviewARGtor import TreeReviewARGtor
    argtor = TreeReviewARGtor(name="debug_tree_review", llm=llm)

    # own
    #from cerg.models.argtors.SystematicARGtor import SystematicARGtor
    #argtor = SystematicARGtor(name="debug_systematic", llm=llm, prompt_type="guided")

    # run experiment
    logging.info("Running debug experiment.")
    res = debug(args.data_path,
                args.venues,
                argtor,
                args.results_dir)

    logging.info("Finished experiment. Results saved at: " + str(res))
