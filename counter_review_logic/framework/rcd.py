import asyncio
import json
import logging
from pathlib import Path

from tqdm import tqdm

from ..data import Review
from . import AutomaticReviewGenerator, AutomaticReviewDataset
from . import PaperCounterfactualDataset


class ReviewDeltaDataset:
    def __init__(self, deltas:dict):
        self._deltas = deltas

    def __iter__(self):
        for pid, deltas in self._deltas.items():
            yield pid, deltas

    def __getitem__(self, item):
        return self._deltas[item]

    def store(self, path: Path, ids_only=True):
        with open(path, "w") as f:
            json.dump({
                pid: {argtor: delta.to_json(ids_only=ids_only) for argtor, delta in deltas.items()} for pid, deltas in self._deltas.items()
            }, f, indent=4)

    @staticmethod
    def load_multi(path:dict[str, str|Path], original_reviews: AutomaticReviewDataset=None, cf_reviews:AutomaticReviewDataset=None):
        # load each individually
        datasets = {}
        for delta_name, delta_path in path.items():
            rds= ReviewDeltaDataset.load(Path(delta_path), original_reviews, cf_reviews)
            datasets[delta_name] = rds

        # aggregate
        all_deltas = {}
        for delta_type, dataset in datasets.items():
            for pid, deltas_by_paper in dataset:
                if pid not in all_deltas:
                    all_deltas[pid] = {}

                for argtor, review_delta in deltas_by_paper.items():
                    if argtor not in all_deltas[pid]:
                        all_deltas[pid][argtor] = {}

                    all_deltas[pid][argtor][delta_type] = review_delta

        # merge
        for pid, deltas_per_argtor in all_deltas.items():
            for argtor, delta_types in deltas_per_argtor.items():

                r1, r2 = None, None
                change_desc = {}
                for delta_type, delta in delta_types.items():
                    r1, r2 = delta.review1, delta.review2

                    change_desc[delta_type] = delta.change

                all_deltas[pid][argtor] = ReviewDelta(r1, r2, change_desc)

        return ReviewDeltaDataset(all_deltas)

    @staticmethod
    def load(path:Path|dict[str,str], original_reviews: AutomaticReviewDataset=None, cf_reviews:AutomaticReviewDataset=None):
        if isinstance(path, dict):
            return ReviewDeltaDataset.load_multi(path, original_reviews, cf_reviews)

        with open(path, "r") as f:
            data = json.load(f)

        deltas = {pid: {argtor: ReviewDelta.from_json(delta) for argtor, delta in deltas.items()}
                       for pid, deltas in data.items()}

        # add reviews by ID if applicable
        if original_reviews is not None and cf_reviews is not None:
            for pid, deltas_by_paper in deltas.items():
                for argtor, delta in deltas_by_paper.items():
                    if "original" in delta.review1.id:
                        assert original_reviews.has_review(delta.review1.id), f"Original review {delta.review1.id} not found in original reviews."
                        assert cf_reviews.has_review(delta.review2.id), f"Counterfactual review {delta.review2.id} not found in counterfactual reviews."
                        delta.review1 = original_reviews.get_review_by_id(delta.review1.id)
                        delta.review2 = cf_reviews.get_review_by_id(delta.review2.id)
                    else:
                        assert original_reviews.has_review(
                            delta.review2.id), f"Original review {delta.review1.id} not found in original reviews."
                        assert cf_reviews.has_review(
                            delta.review1.id), f"Counterfactual review {delta.review2.id} not found in counterfactual reviews."
                        delta.review2 = original_reviews.get_review_by_id(delta.review2.id)
                        delta.review1 = cf_reviews.get_review_by_id(delta.review1.id)

        return ReviewDeltaDataset(deltas)


class ReviewDelta:
    def __init__(self, review1:Review, review2:Review, change_desc:dict):
        self._review1 = review1
        self._review2 = review2
        self._change = change_desc

    @property
    def review1(self):
        return self._review1

    @review1.setter
    def review1(self, value):
        self._review1 = value

    @property
    def review2(self):
        return self._review2

    @review2.setter
    def review2(self, value):
        self._review2 = value

    @property
    def change(self):
        return self._change

    @change.setter
    def change(self, value):
        self._change = value

    def to_json(self, ids_only):
        return {
            "review1": self._review1.to_json(id_only=ids_only),
            "review2": self._review2.to_json(id_only=ids_only),
            "change": self._change
        }

    @staticmethod
    def from_json(obj):
        return ReviewDelta(Review.from_json(obj["review1"]), Review.from_json(obj["review2"]), obj["change"])


class ReviewChangeDetector:
    def __init__(self, name):
        self.name = name

    def run(self, review1, review2, **config) -> ReviewDelta:
        pass

    async def arun(self, review1, review2, **config) -> ReviewDelta:
        pass


class ReviewChangeDetectionPipeline:
    def __init__(self,
                 argtors:list[AutomaticReviewGenerator]|list[str],
                 cfd:PaperCounterfactualDataset,
                 cf_reviews:AutomaticReviewDataset,
                 o_reviews:AutomaticReviewDataset,
                 disk_cache_dir:str|Path,
                 run_asnyc=False):
        self.argtors = []
        for a in argtors:
            self.argtors += [a.name] if isinstance(a, AutomaticReviewGenerator) else [a]

        self.paper_counterfactual_dataset = cfd
        self.reviews_for_cfpapers = cf_reviews
        self.reviews_for_opapers = o_reviews

        self.disk_cache_dir = Path(disk_cache_dir)

        self.cached = {}

    def store(self, path: Path):
        ReviewDeltaDataset(self.cached).store(path, ids_only=True)

    def load(self, path:Path, original_reviews:AutomaticReviewDataset, counterfactual_reviews:AutomaticReviewDataset):
        self.cached = ReviewDeltaDataset.load(path, original_reviews, counterfactual_reviews)._deltas

    def deltas_for_all(self, review_change_detector:ReviewChangeDetector,  **config):
        result = []
        for paper_id, paper, cf in self.paper_counterfactual_dataset:
            argtor_names = [a for a in self.argtors]

            original_reviews = [self.reviews_for_opapers.get_review(paper_id, an) for an in argtor_names]
            cf_reviews = [self.reviews_for_cfpapers.get_review(paper_id, an) for an in argtor_names]

            # compute deltas between original reviews and the counterfactual ones
            items_to_compute = []
            for argtor_name, cfrev, orev in zip(argtor_names, cf_reviews, original_reviews):
                if cfrev is None or orev is None:
                    logging.info("Skipping delta computation for paper %s and argtor %s due to missing reviews.", paper_id, argtor_name)
                    continue

                if paper_id not in self.cached or argtor_name not in self.cached[paper_id]:
                    items_to_compute += [(orev, cfrev, argtor_name, paper_id)]

            def compute_delta(orev, cfrev, argtor_name, paper_id):
                print("Computing delta", orev.id, cfrev.id, argtor_name, paper_id)
                delta = review_change_detector.run(orev, cfrev, **config)
                return paper_id, argtor_name, delta

            deltas = [compute_delta(orev, cfrev, argtor_name, paper_id) for orev, cfrev, argtor_name, paper_id in items_to_compute]

            for paper_id, argtor_name, delta in deltas:
                if delta is None:
                    logging.warning("Delta computation for paper %s and argtor %s returned None.", paper_id, argtor_name)
                    continue

                to_save = False
                if paper_id not in self.cached:
                    to_save = True
                    self.cached[paper_id] = {}

                self.cached[paper_id][argtor_name] = delta

                result += [self.cached[paper_id][argtor_name]]

                if to_save:
                    self.store(self.disk_cache_dir)

    async def adeltas_for_all(self, review_change_detector:ReviewChangeDetector,  **config):
        self.cache_lock = asyncio.Lock()

        result = []
        for paper_id, paper, cf in tqdm(self.paper_counterfactual_dataset, desc="Computing deltas for all papers"):
            argtor_names = [a for a in self.argtors]

            original_reviews = [self.reviews_for_opapers.get_review(paper_id, an) for an in argtor_names]
            cf_reviews = [self.reviews_for_cfpapers.get_review(paper_id, an) for an in argtor_names]

            # compute deltas between original reviews and the counterfactual ones
            items_to_compute = []
            for argtor_name, cfrev, orev in zip(argtor_names, cf_reviews, original_reviews):
                if cfrev is None or orev is None:
                    logging.info("Skipping delta computation for paper %s and argtor %s due to missing reviews.", paper_id, argtor_name)
                    continue

                if paper_id not in self.cached or argtor_name not in self.cached[paper_id]:
                    items_to_compute += [(orev, cfrev, argtor_name, paper_id)]

            if len(items_to_compute) == 0:
                continue

            async def compute_delta(orev, cfrev, argtor_name, paper_id):
                logging.info(f"Computing delta: {argtor_name}, {paper_id}, {cfrev.id}")
                delta = await review_change_detector.arun(orev, cfrev, **config)
                return paper_id, argtor_name, delta

            tasks = [compute_delta(orev, cfrev, argtor_name, paper_id) for orev, cfrev, argtor_name, paper_id in items_to_compute]
            deltas = await asyncio.gather(*tasks)

            async with self.cache_lock:
                for paper_id, argtor_name, delta in deltas:
                    if delta is None:
                        logging.warning("Delta computation for paper %s and argtor %s returned None.", paper_id, argtor_name)
                        continue

                    if paper_id not in self.cached:
                        self.cached[paper_id] = {}

                    self.cached[paper_id][argtor_name] = delta

                    result += [self.cached[paper_id][argtor_name]]

                    self.store(self.disk_cache_dir)