import copy
import json
import logging
from json import JSONDecodeError
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from ..data import Paper, Review


class AutomaticReviewGenerator:
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _load_prompt(fp):
        with open(fp, "r") as f:
            msgs = json.load(f)

        tmplt = []
        for m in msgs:
            assert m["actor"] in ["human", "system"]
            tmplt += [(m["actor"], m["msg"])]

        return ChatPromptTemplate.from_messages(tmplt)

    @staticmethod
    def _parse_as_review(output:str, venue_config:dict):
        """
        Parses the output of the model into a review object.
        :param output: the output of the model
        :param venue_config: the venue config
        :return: a review object
        """
        raise NotImplementedError("This method needs to be implemented in the subclass.")

    def run(self, paper:Paper, **config) -> Review:
        pass


class AutomaticReviewDataset:
    def __init__(self, reviews: dict[str, dict[str, Review]] = None):
        self._reviews = reviews if reviews else {}
        self._review_index = {}

        self._index()

    def _index(self):
        self._review_index = {rev.id: (pid, argtor) for pid, revs in self._reviews.items() for argtor, rev in revs.items()}

    def set_review(self, pid, argtor_name, review):
        if pid not in self._reviews:
            self._reviews[pid] = {}

        self._reviews[pid][argtor_name] = review
        self._review_index[review.id] = (pid, argtor_name)

    def get_review(self, pid, argtor_name):
        return self._reviews[pid][argtor_name] if pid in self._reviews and argtor_name in self._reviews[pid] else None

    def get_review_by_id(self, id):
        if id not in self._review_index:
            raise KeyError("Provided review id is not in the dataset.")

        return self._reviews[self._review_index[id][0]][self._review_index[id][1]]

    def has_review(self, id):
        return id in self._review_index

    def get_reviews(self, pid):
        return self._reviews[pid]

    def get_pids(self):
        return list(self._reviews.keys())

    def get_argtors(self):
        return list(set(k for v in self._reviews.values() for k in v.keys()))

    def get_reviews_per_argtor(self, argtor_name):
        return {pid: rev[argtor_name] for pid, rev in self._reviews.items() if argtor_name in rev}

    def __contains__(self, key):
        return key in self._reviews

    def __len__(self):
        return len(self._reviews)

    def __iter__(self):
        return iter(self._reviews)

    @staticmethod
    def load(inp: str | Path):
        """
        Loads dataset from disk or from the provided sample cache dict.
        :param inp:
        :return:
        """
        if type(inp) == str:
            inp = Path(inp)

        assert inp.exists(), f"provided path for loading does not exist: {inp}"
        assert inp.is_dir(), f"provided path should be a dir: {inp}"

        res = {}
        for argtor_dir in inp.iterdir():
            if not argtor_dir.is_dir():
                continue

            argtor_name = argtor_dir.name

            for file in argtor_dir.iterdir():
                if file.suffix != ".json":
                    continue

                pid = file.stem

                try:
                    with file.open("r") as f:
                        o = Review.from_json(json.load(f))
                except JSONDecodeError as e:
                    logging.error("Failed to load review from file: %s. Error: %s", file, e)
                    continue

                if pid not in res:
                    res[pid] = {}

                res[pid][argtor_name] = o

        _reviews = res

        if len(_reviews) == 0:
            logging.warning(f"Loading form a directory without reviews: {inp}")

        return AutomaticReviewDataset(_reviews)

    def store(self, fp: str | Path):
        """
        Stores the dataset for later reuse. Stored as JSON.

        :param fp: the file path
        :return: None
        """
        if type(fp) == str:
            fp = Path(fp)

        assert fp.is_dir(), "the provided filepath should point to a directory"
        assert fp.exists(), "the provided directory has to exist"

        for pid, argtors in self._reviews.items():
            for argtor_name, result in argtors.items():
                target_path = fp / argtor_name
                target_path.mkdir(exist_ok=True, parents=False)

                target_file = target_path/ f"{pid}.json"

                # we override the file if it exists!

                with target_file.open("w+") as f:
                    json.dump(result.to_json(), f, indent=4)


class AutomaticReviewGenerationPipeline:
    def __init__(self, dataset:list[Paper], review_id_prefix:str):
        self.cached_samples = AutomaticReviewDataset()
        self.review_id_prefix = review_id_prefix

        self.dataset = {p.id: p for p in dataset}

    def load_samples(self, fp: str | Path):
        """
        Loads the cached samples from a file.

        :param fp: the file path
        :return: None
        """
        self.cached_samples = AutomaticReviewDataset.load(fp)

    def store_samples(self, fp: str | Path):
        """
        Stores the cached samples to a file.

        :param fp: the file path
        :return: None
        """
        self.cached_samples.store(fp)

    def copy_samples(self, other):
        """
        Given another eval pipeline, it copies the cached responses for further processing.

        :param other:
        :return:
        """
        self.cached_samples = copy.deepcopy(other.cached_samples)

    def sample_for_paper(self,
                         pid: str,
                         argtors: AutomaticReviewGenerator | list[AutomaticReviewGenerator],
                         **argtor_config) -> Review|list[Review]:
        """
        Produces a set of reviews for the given automatic review generators on the provided paper.

        :param paper: the paper to generate reviews for
        :param argtors: the automatic review generators
        :param argtor_config: the configuration for the automatic review generators
        """
        assert pid in self.dataset, f"Provided paper id {pid} is not in the dataset."
        paper = self.dataset[pid]

        if type(argtors) != list:
            argtors = [argtors]

        # generate reviews "repeat"-times for the original o_paper
        reviews = []
        for argtor in argtors:
            if pid in self.cached_samples and argtor.name in self.cached_samples.get_reviews(pid):
                reviews += [self.cached_samples.get_review(pid, argtor.name)]
            else:
                review = argtor.run(paper, **argtor_config)

                if review is None:
                    reviews += [None]
                    continue

                review.id = f"{self.review_id_prefix}_{argtor.name}_{pid}"

                reviews += [review]

        # cache sample
        for i, argtor in enumerate(argtors):
            if reviews[i] is not None:
                self.cached_samples.set_review(paper.id, argtor.name, reviews[i])

        if len(argtors) == 1:
            return reviews[0]
        else:
            return reviews

    def sample_all(self, argtors:list[AutomaticReviewGenerator], **config) -> tuple[str, list[Review]]:
        """
        This samples reviews for all papers and their counterfactuals of the dataset. This function returns a generator.
        The reviews generated throughout are stored in the cache of this class to be used for evaluation later on.

        :param argtors: list of review generators
        :param config: the config
        :return: a generator of tuples where the first entry is the id of the o_paper, the reviews for the original o_paper and the reviews for the counterfactuals
        """
        for pid in tqdm(self.dataset, desc="Sampling reviews for papers"):
            yield pid, self.sample_for_paper(pid, argtors, **config)

    def batch_all(self, argtors:list[AutomaticReviewGenerator], **config) -> tuple[str, list[Review]]:
        """
        This samples reviews for all papers and their counterfactuals of the dataset. This function returns a generator.
        The reviews generated throughout are stored in the cache of this class to be used for evaluation later on.

        :param argtors: list of review generators
        :param config: the config
        :return: a generator of tuples where the first entry is the id of the o_paper, the reviews for the original o_paper and the reviews for the counterfactuals
        """
        for pid in tqdm(self.dataset, desc="Sampling reviews for papers"):
            yield pid, self.sample_for_paper(pid, argtors, **config)