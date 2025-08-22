import json
import logging
from pathlib import Path

from . import AutomaticReviewGenerator
from . import PaperCounterfactualDataset
from . import ReviewDelta, ReviewDeltaDataset


class ReviewDeltaEvaluator:
    def __init__(self, name):
        self.name = name

    def run(self, rd: ReviewDelta | list[ReviewDelta], **config) -> dict:
        pass


class ReviewDeltaEvaluationDataset:
    def __init__(self, eval_data: dict[str, dict[str, list[dict]]]):
        self.data = eval_data

    def store(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=4)

    @staticmethod
    def load(path: Path):
        with open(path, "r") as f:
            data = json.load(f)

        return ReviewDeltaEvaluationDataset(data)

    def __getitem__(self, item):
        return self.data[item]


class ReviewDeltaEvaluatorPipeline:
    def __init__(self,
                 rcd_dataset: ReviewDeltaDataset,
                 paper_counterfactual_dataset: PaperCounterfactualDataset,
                 rde: ReviewDeltaEvaluator,
                 disk_caching_path: Path = None):
        self.deltas = rcd_dataset
        self.cf_paper_dataset = paper_counterfactual_dataset

        self.evaluator = rde

        self.cached = {}

    def store(self, path: Path):
        ReviewDeltaEvaluationDataset(self.cached).store(path)

    def load(self, path: Path):
        self.cached = ReviewDeltaEvaluationDataset.load(path).data

    def run(self, argtors: list[AutomaticReviewGenerator | str], **config) -> dict[str, dict[str, dict[str, dict]]]:
        result = self.cached if self.cached is not None else {}

        # turn argtors into a list of names
        n_argtors = []
        for argtor in argtors:
            if isinstance(argtor, str):
                n_argtors.append(argtor)
            elif isinstance(argtor, AutomaticReviewGenerator):
                n_argtors.append(argtor.name)
            else:
                raise ValueError("Invalid argtor type: {}".format(type(argtor)))
        argtors = n_argtors

        # get the deltas for each paper and all its counterfactuals
        deltas = {argtor: [] for argtor in argtors}
        for paper_id, delta in self.deltas:
            for argtor in argtors:
                if argtor not in delta:
                    logging.warning("Expected argtor {} in deltas for paper {} but did not find it.".format(argtor, paper_id))
                    continue

                deltas[argtor].append(delta[argtor])

        # evaluate the deltas for each argtor
        for argtor in deltas:
            if argtor in result and len(result[argtor]) > 0: # already covered
                continue

            result[argtor] = self.evaluator.run(deltas[argtor], **config)

        return result
