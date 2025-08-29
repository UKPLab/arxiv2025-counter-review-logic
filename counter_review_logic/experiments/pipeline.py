import json
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from ..data import Paper
from ..framework import AutomaticReviewGenerator, AutomaticReviewGenerationPipeline, AutomaticReviewDataset, \
    PaperCounterfactualGenerator, PaperCounterfactualDataset, ReviewDeltaEvaluator, ReviewDeltaEvaluatorPipeline, \
    ReviewChangeDetector, ReviewDeltaDataset, ReviewChangeDetectionPipeline


def create_cf_dataset(paper_dataset: list[Paper],
                      cf_generators: list[PaperCounterfactualGenerator] | PaperCounterfactualGenerator,
                      out_path: Path,
                      cached: bool,
                      **cf_config) -> list[Path]:
    """
    This creates a counterfactual dataset for the given papers based on the cf generators.

    :param paper_dataset: the papers to generate counterfactuals for
    :param paper_dataset_name:  the name of the dataset (will be used for storing)
    :param cf_generators: the counterfactual generators
    :param out_path: the output path
    :param cached: whether to use cached data
    :param cf_config: the configuration for the counterfactual generators
    :return:
    """
    assert out_path.exists(), f"output path has to exist. {out_path} does not."

    if not isinstance(cf_generators, list):
        cf_generators = [cf_generators]

    if not cached:
        logging.warning("You are running the pipeline without caching. This will overwrite existing data.")

    res = []
    for cfgen in tqdm(cf_generators, desc="Creating dataset iterating over CF Generators"):
        cf_name = cfgen.name
        target_path = out_path / cf_name

        if not cached and target_path.exists():
            logging.info(f"Deleting existing dataset {target_path}")
            shutil.rmtree(target_path, ignore_errors=True)

        # create dir
        target_path.mkdir(parents=False, exist_ok=True)

        # use caching
        cfgen.set_disk_cache(target_path, paper_dataset)

        # run generator
        cf_data = cfgen.run(paper_dataset, **cf_config)
        cf_data.save(target_path, force=not cached)

        res += [target_path]

    return res


def generate_reviews(papers: list[Paper],
                     dataset_name: str,
                     argtors: list[AutomaticReviewGenerator] | AutomaticReviewGenerator,
                     out_path: Path,
                     cached: bool,
                     argtor_config: dict | None = None) -> Path:
    assert out_path.exists(), f"output path has to exist. {out_path} does not."

    if not isinstance(argtors, list):
        argtors = [argtors]

    # check target dir
    target_path = out_path / dataset_name
    load_existing = cached and target_path.exists()

    if load_existing:
        logging.info("Found existing data. Will only process missing papers and argtors.")

    # actually generate the reviews if no data yet
    logging.info("Creating new reviews for the dataset.")

    # if needed create path
    target_path.mkdir(parents=False, exist_ok=True)

    pipe = AutomaticReviewGenerationPipeline(papers, review_id_prefix=dataset_name)

    # create pipeline for remaining papers
    if load_existing:
        pipe.load_samples(target_path)

    # generate reviews on dataset
    cnt = 0
    for s in tqdm(pipe.sample_all([argtor for argtor in argtors],
                                  **argtor_config if argtor_config else {}), desc="Generating reviews per paper"):
        cnt += 1

        # cache files on disc
        if cnt % 10 == 0:
            pipe.store_samples(target_path)

    pipe.store_samples(target_path)

    return target_path


def generate_reviews_for_counterfactuals(original_papers: list[Paper],
                                         cf_dataset_path: Path,
                                         argtors: list[AutomaticReviewGenerator] | AutomaticReviewGenerator,
                                         out_path: Path,
                                         cached: bool,
                                         argtor_config: dict | None = None) -> Path:
    assert out_path.exists(), f"output path has to exist. {out_path} does not."

    if not isinstance(argtors, list):
        argtors = [argtors]

    # load counterfactual dataset
    cf_dataset = PaperCounterfactualDataset.load(cf_dataset_path, originals=original_papers)

    # generate reviews
    return generate_reviews([cf.cf_paper for _, _, cf in cf_dataset], cf_dataset.meta["counterfactual_type"], argtors,
                            out_path, cached, argtor_config)


def generate_reviews_for_originals(dataset: list[Paper],
                                   argtors: list[AutomaticReviewGenerator] | AutomaticReviewGenerator,
                                   out_path: Path,
                                   cached: bool,
                                   argtor_config: dict | None = None) -> Path:
    """
    The out_path should point to the reviews dir. if there is already a dataset, it will be loaded on
    the subpath "original".

    :param dataset:
    :param argtors:
    :param out_path:
    :param argtor_config:
    :return:
    """
    return generate_reviews(dataset, "original", argtors, out_path, cached, argtor_config)


def determine_review_deltas(original_papers: list[Paper],
                            cf_review_dataset_path: Path,
                            original_review_dataset_path: Path,
                            argtors: list[AutomaticReviewGenerator] | AutomaticReviewGenerator | list[str] | str,
                            cf_paper_dataset_path: Path,
                            rcd: ReviewChangeDetector,
                            out_path: Path,
                            cached: bool,
                            config: dict | None = None):
    assert out_path.exists(), f"output path has to exist. {out_path} does not."

    if not isinstance(argtors, list):
        argtors = [argtors]

    # load cf dataset
    cf_dataset = PaperCounterfactualDataset.load(cf_paper_dataset_path, originals=original_papers)

    # set target path
    target_path = out_path / cf_dataset.meta["counterfactual_type"]
    target_path.mkdir(parents=False, exist_ok=True)

    target_path = target_path / f"{rcd.name}_deltas.json"

    # gather input reviews
    original_reviews = AutomaticReviewDataset.load(original_review_dataset_path)
    cf_reviews = AutomaticReviewDataset.load(cf_review_dataset_path)

    # load cached if applicable
    if target_path.exists() and cached:
        print("Loading detlas from", target_path)

        pipe = ReviewChangeDetectionPipeline(argtors, cf_dataset, cf_reviews, original_reviews,
                                             disk_cache_dir=target_path)
        pipe.load(target_path, original_reviews, cf_reviews)

        print("Loaded cached deltas from", target_path, "with", len(pipe.cached), "entries.")

        # check if deltas for all argtors and papers are present (otherwise run again)
        if all([pipe.cached.get(p, {}).get(a, None) for p in cf_dataset.get_pids() for a in argtors]):
            return target_path

    # run pipeline (for missing elements)
    if config is None:
        config = {}

    pipe = ReviewChangeDetectionPipeline(argtors, cf_dataset, cf_reviews, original_reviews, disk_cache_dir=target_path)
    pipe.deltas_for_all(rcd, **config)

    pipe.store(target_path)

    return target_path


async def adetermine_review_deltas(original_papers: list[Paper],
                                   cf_review_dataset_path: Path,
                                   original_review_dataset_path: Path,
                                   argtors: list[AutomaticReviewGenerator] | AutomaticReviewGenerator | list[str] | str,
                                   cf_paper_dataset_path: Path,
                                   rcd: ReviewChangeDetector,
                                   out_path: Path,
                                   cached: bool,
                                   config: dict | None = None):
    assert out_path.exists(), f"output path has to exist. {out_path} does not."

    if not isinstance(argtors, list):
        argtors = [argtors]

    # load cf dataset
    cf_dataset = PaperCounterfactualDataset.load(cf_paper_dataset_path, originals=original_papers)

    # set target path
    target_path = out_path / cf_dataset.meta["counterfactual_type"]
    target_path.mkdir(parents=False, exist_ok=True)

    target_path = target_path / f"{rcd.name}_deltas.json"

    # gather input reviews
    original_reviews = AutomaticReviewDataset.load(original_review_dataset_path)
    cf_reviews = AutomaticReviewDataset.load(cf_review_dataset_path)

    # load cached if applicable
    if target_path.exists() and cached:
        print("Loading detlas from", target_path)

        pipe = ReviewChangeDetectionPipeline(argtors, cf_dataset, cf_reviews, original_reviews,
                                             disk_cache_dir=target_path)
        pipe.load(target_path, original_reviews, cf_reviews)

        print("Loaded cached deltas from", target_path, "with", len(pipe.cached), "entries.")

        # check if deltas for all argtors and papers are present (otherwise run again)
        if all([pipe.cached.get(p, {}).get(a, None) for p in cf_dataset.get_pids() for a in argtors]):
            return target_path
    else:
        pipe = ReviewChangeDetectionPipeline(argtors, cf_dataset, cf_reviews, original_reviews,
                                             disk_cache_dir=target_path)

    # run pipeline (for missing elements)
    if config is None:
        config = {}

    await pipe.adeltas_for_all(rcd, **config)

    pipe.store(target_path)

    return target_path


def evaluate_review_deltas(
        original_papers: list[Paper],
        review_deltas_dataset_path: Path,
        cf_paper_dataset_path: Path,
        argtors: list[AutomaticReviewGenerator] | AutomaticReviewGenerator,
        rcd: ReviewChangeDetector | str,
        rde: ReviewDeltaEvaluator,
        out_path: Path,
        cached: bool,
        config: dict | None = None):
    assert out_path.exists(), f"output path has to exist. {out_path} does not."
    assert rde is not None, "RDE needs to be provided"
    assert rcd is not None, "RCD needs to be provided"

    if isinstance(rcd, ReviewChangeDetector):
        rcd = rcd.name

    if not isinstance(argtors, list):
        argtors = [argtors]

    # load cf dataset
    cf_dataset = PaperCounterfactualDataset.load(cf_paper_dataset_path, originals=original_papers)

    # load deltas
    delta_dataset = ReviewDeltaDataset.load(review_deltas_dataset_path)

    # set target path
    target_path = out_path / cf_dataset.meta["counterfactual_type"]
    target_path.mkdir(parents=False, exist_ok=True)

    target_path = target_path / f"{cf_dataset.meta['counterfactual_type']}__{rcd}__{rde.name}_eval.json"

    # create pipeline
    pipe = ReviewDeltaEvaluatorPipeline(delta_dataset, cf_dataset, rde)

    # load pre-existing evaluations
    if target_path.exists() and cached:
        pipe.load(target_path)

    # run evaluation on missing examples and store result
    if config is None:
        config = {}

    pipe.run(argtors, **config)
    pipe.store(target_path)

    return target_path


def create_file_structure(out_path, experiment_name):
    assert out_path.exists(), f"output path has to exist. {out_path} does not."

    out_path = out_path / experiment_name
    out_path.mkdir(parents=False, exist_ok=True)

    reviews_dir = out_path / "reviews"
    reviews_dir.mkdir(parents=False, exist_ok=True)

    delta_dir = out_path / "deltas"
    delta_dir.mkdir(parents=False, exist_ok=True)

    result_dir = out_path / "eval"
    result_dir.mkdir(parents=False, exist_ok=True)

    dataset_dir = out_path / "cf_datasets"
    dataset_dir.mkdir(parents=False, exist_ok=True)

    return out_path, reviews_dir, delta_dir, result_dir, dataset_dir


def full_pipeline(paper_dataset: list[Paper],
                  paper_dataset_name: str,
                  cf_generators: list[PaperCounterfactualGenerator] | PaperCounterfactualGenerator | None,
                  out_path: str | Path,
                  experiment_name: str,
                  argtors: list[AutomaticReviewGenerator] | AutomaticReviewGenerator | None,
                  rcd: ReviewChangeDetector | None,
                  rde: ReviewDeltaEvaluator | None,
                  argtor_config: dict | None = None,
                  rcd_config: dict | None = None,
                  cached=True,
                  o_reviews_path: Path | None = None,
                  ):
    """
    This is the full pipeline for the CERG framework. It generates counterfactual papers, reviews for the original
    papers, reviews for the counterfactual papers, detects deltas between the reviews, and evaluates the deltas.

    The output folder structure looks as follows:
    > experiment name
        > cf_datasets
            > [cf name]
        > reviews
            > original
            > [cf gen names]
        > deltas
            > [rcd name]
                > [cf name]
        > eval
            > [cf name]
                > [cf name]_[rcd name]_[rde name]

    :param paper_dataset:  the papers to generate counterfactuals for
    :param paper_dataset_name:  the name of the dataset (will be used for storing)
    :param cf_generators:  the counterfactual generators
    :param out_path:  the output path (top level dir)
    :param experiment_name:  the name of the experiment (will be used for storing
    :param argtors: the review generators
    :param rcd: the review change detector to use
    :param rde: the review delta evaluator to use
    :param argtor_config: the configuration for the review generators
    :param rcd_config: the configuration for the review change detector
    :param cached: if set to True, present data will be used if available; set to False to force a full override
    :param o_reviews_path: the path to the original reviews (if not provided, they will be generated)
    :return:    the paths to the evaluation results
    """

    if isinstance(out_path, str):
        out_path = Path(out_path)

    # prepping
    if argtor_config is None and argtors is not None and len(argtors) > 0:
        logging.warning("argtor_config is not set although you passed argtors for generation.")

    if cached:
        logging.warning(
            "You are running the pipeline with cached data. Set cached to False to overwrite existing data.")

    # create file structure
    out_path, reviews_dir, delta_dir, result_dir, dataset_dir = create_file_structure(out_path, experiment_name)

    # save covered paper dataset
    if (out_path / "meta.json").exists():
        meta = json.load(open(out_path / "meta.json"))
    else:
        meta = {"dataset_name": paper_dataset_name, "papers": []}

    meta["papers"] += [p.id for p in paper_dataset]

    with open(out_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    # 1. create counterfactual papers
    logging.info("=== CREATING COUNTERFACTUALS" + "=" * 10)
    cf_dataset_paths = create_cf_dataset(paper_dataset, cf_generators, dataset_dir, cached)

    # 2. run review generators on original papers (or simply provide the file link)
    if o_reviews_path is None:
        logging.info("=== GENERATING REVIEWS FOR ORIGINAL PAPERS" + "=" * 10)

        o_reviews_path = reviews_dir
        o_reviews_path = generate_reviews_for_originals(paper_dataset, argtors, o_reviews_path, cached, argtor_config)
    else:
        assert o_reviews_path.exists(), f"Provided original reviews path {o_reviews_path} does not exist."

        data = AutomaticReviewDataset.load(o_reviews_path)

        assert set(data.get_pids()) == set(
            p.id for p in paper_dataset), "Provided original reviews do not match the dataset."

    # 3. run review generators on counterfactual papers
    logging.info("=== GENERATING REVIEWS FOR CF PAPERS" + "=" * 10)

    review_paths = []
    for dp in cf_dataset_paths:
        review_paths += [
            generate_reviews_for_counterfactuals(paper_dataset, dp, argtors, reviews_dir, cached, argtor_config)]

    # 4. run delta-detection on original and counterfactual reviews
    logging.info("=== DETECTING DELTAS" + "=" * 10)

    delta_paths = []
    for rp, dp in zip(review_paths, cf_dataset_paths):
        delta_paths += [
            determine_review_deltas(paper_dataset, rp, o_reviews_path, argtors, dp, rcd, delta_dir, cached, rcd_config)]

    # 5. evaluate the deltas
    logging.info("=== EVALUATING REVIEWS" + "=" * 10)
    eval_paths = []
    for reviews_path, cf_path, delta_path in zip(review_paths, cf_dataset_paths, delta_paths):
        eval_paths += [evaluate_review_deltas(paper_dataset, delta_path, cf_path, argtors, rcd, rde, result_dir, cached,
                                              rcd_config)]

    return eval_paths
