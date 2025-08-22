import copy
import json
import logging
import os
import random
from pathlib import Path

from tqdm import tqdm

from ..data import Paper


class PaperCounterfactual:
    def __init__(self, o_paper: Paper, counterfactual_type: str, cf_paper: Paper, changes: dict):
        self._o_paper = o_paper
        self._cf_paper = cf_paper

        self._changes = changes
        self._counterfactual_type = counterfactual_type

    @property
    def changes(self):
        return self._changes

    @changes.setter
    def changes(self, changes: dict):
        self._changes = changes

    @property
    def ctype(self):
        return self._counterfactual_type

    @property
    def o_paper(self):
        return self._o_paper

    @property
    def cf_paper(self):
        return self._cf_paper

    def to_json_obj(self):
        return {
            "o_paper": self._o_paper.id,
            "cf_paper": self._cf_paper.to_json_obj(),
            "changes": self._changes,
            "counterfactual_type": self._counterfactual_type
        }

    @staticmethod
    def from_json_obj(obj, opaper):
        assert obj["o_paper"] == opaper.id, "the provided original o_paper does not match the counterfactual to load"

        return PaperCounterfactual(
            o_paper=opaper,
            counterfactual_type=obj["counterfactual_type"],
            cf_paper=Paper.from_json_obj(obj["cf_paper"]),
            changes=obj["changes"]
        )


class PaperCounterfactualDataset:
    def __init__(self,
                 original: dict[str, Paper],
                 counterfactuals: dict[str, PaperCounterfactual] | None = None,
                 counterfatual_type: str | None = None,
                 meta: dict | None = None):
        self.original_papers = None
        self.counterfactual_papers = None
        self.meta = meta if meta is not None else {}
        self.pid_index = None

        self.meta["counterfactual_type"] = counterfatual_type if counterfatual_type is not None else self.meta.get("counterfactual_type", "unknown")

        if counterfactuals is None:
            counterfactuals = {}

        self.setup(original, counterfactuals)

    def setup(self, original: dict, counterfactuals: dict[str, PaperCounterfactual]):
        if len(counterfactuals.keys()) == 0 or set(original.keys()).issubset(set(counterfactuals.keys())):
            logging.warning("the provided data does not provide counterfactuals for all original papers")

        self.original_papers = original
        self.counterfactual_papers = counterfactuals if len(counterfactuals.keys()) > 0 else {pid: None for pid in
                                                                                              self.original_papers}

        self.pid_index = list(self.original_papers.keys())

    def shuffle(self):
        random.shuffle(self.pid_index)

    def get_pids(self):
        return self.pid_index

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.original_papers
        elif isinstance(item, int):
            return item in self.pid_index
        else:
            raise ValueError(f"Cannot check if dataset contains {item}. Use string or integer.")

    def __iter__(self) -> tuple[str, Paper, PaperCounterfactual]:
        for paper_id in self.pid_index:
            paper = self.original_papers[paper_id]
            cf = self.counterfactual_papers[paper_id]

            yield paper_id, paper, cf

    def __getitem__(self, item: str | int | tuple[int, int]):
        if isinstance(item, int):
            item = self.pid_index[item]
            return self.original_papers[item], self.counterfactual_papers[item]
        elif isinstance(item, str):
            return self.original_papers[item], self.counterfactual_papers[item]
        elif isinstance(item, slice):
            items = self.pid_index[item]
            return PaperCounterfactualDataset(original={k: v for k, v in self.original_papers.items() if k in items},
                                              counterfactuals={k: v for k, v in self.counterfactual_papers.items() if
                                                               k in items},
                                              meta={"subset": items, **self.meta}
                                              )
        else:
            raise ValueError(f"Cannot access dataset by {type(item)} use string or integer or integer range.")

    def __setitem__(self, key: str, value: tuple[Paper, list[PaperCounterfactual]]):
        if key not in self.original_papers:
            self.pid_index += [key]

        self.original_papers[key] = value[0]
        self.counterfactual_papers[key] = value[1]

    def __len__(self):
        return len(self.original_papers)

    def save(self, fp: str|Path, force=False):
        if isinstance(fp, str):
            fp = Path(fp)

        for pid in self.original_papers:
            tp = fp / f"{pid}.json"
            if tp.exists() and not force:
                logging.debug(f"File {tp} already exists, skipping...")
                continue

            if pid not in self.counterfactual_papers:
                logging.warning(f"Paper {pid} does not have a counterfactual, skipping...")
                continue

            cf = self.counterfactual_papers[pid]

            with tp.open("w+") as f:
                json.dump(cf.to_json_obj(), f, indent=4)

        tp = fp / "meta.json"
        with tp.open("w+") as f:
            json.dump(self.meta, f, indent=4)

    @staticmethod
    def load(fp: str|Path, originals: list[Paper]):
        if isinstance(fp, str):
            fp = Path(fp)

        if not fp.exists():
            raise ValueError(f"Provided filepath {fp} does not exist. Cannot load data.")

        originals = {p.id: p for p in originals}

        original_papers = {}
        counterfactual_papers = {}

        meta = {}
        for file in fp.iterdir():
            if file.name == "meta.json":
                with file.open("r") as f:
                    meta = json.load(f)
            else:
                with file.open("r") as f:
                    data = json.load(f)

                if data["o_paper"] not in originals:
                    logging.warning(f"Original paper {data['o_paper']} not found in provided originals, skipping...")
                    continue

                original = originals[data["o_paper"]]

                cf = PaperCounterfactual.from_json_obj(data, original)

                original_papers[original.id] = original
                counterfactual_papers[original.id] = cf

        return PaperCounterfactualDataset(
            meta=meta,
            original=original_papers,
            counterfactuals=counterfactual_papers
        )

    def save_json(self, fp: str, force=False):
        if isinstance(fp, str):
            fp = Path(fp)

        if not fp.suffix.endswith(".json"):
            fp = fp.with_suffix(fp.suffix + ".json")

        if fp.exists() and not force:
            raise ValueError(f"Filepath {fp.absolute()} already exists. Erase or pass force flag to override.")

        o_papers = {pid: p.to_json_obj() for pid, p in self.original_papers.items()}
        cf_papers = {pid: p.to_json_obj() for pid, p in self.counterfactual_papers.items()}

        with open(fp, "w+") as f:
            json.dump({
                "meta": self.meta,
                "original_papers": o_papers,
                "counterfactual_papers": cf_papers,
            }, f, indent=4)

    @staticmethod
    def load_json(fp: str|Path):
        if isinstance(fp, str):
            fp = Path(fp)

        if not fp.exists():
            raise ValueError(f"Provided filepath {fp} does not exist. Cannot load data.")

        with fp.open("r") as f:
            raw = json.load(f)

        original_papers = {pid: Paper.from_json_obj(obj) for pid, obj in raw["original_papers"].items()}

        return PaperCounterfactualDataset(
            meta=raw["meta"],
            original=original_papers,
            counterfactuals={pid: PaperCounterfactual.from_json_obj(obj, original_papers[pid]) for pid, obj in
                             raw["counterfactual_papers"].items()}
        )

    def __str__(self):
        return str({
            "meta": str(self.meta),
            "#original_papers": len(self.original_papers),
            "counterfactual_types": str(set(cf.ctype for cf in self.counterfactual_papers.values())),
            "original_papers": str(list(self.original_papers.keys()))
        })


class PaperCounterfactualGenerator:
    def __init__(self, name: str):
        self.name = name

        self._disk_cache_dir = None
        self._disk_cache_freq = 1
        self._cache = None

    def single_run(self, paper: Paper, **config) -> PaperCounterfactual:
        raise NotImplementedError("Not implemented in base class. Extend it and override method!")

    def set_disk_cache(self, cache_dir:str, papers:list[Paper]):
        self._disk_cache_dir = cache_dir
        self._cache = PaperCounterfactualDataset.load(cache_dir, originals=papers)

        logging.info(f"Loaded CFGEn disk cache from {cache_dir} with {len(self._cache)} papers.")

    def run(self, papers: list[Paper], **config) -> PaperCounterfactualDataset:
        res = {}
        errors = {}
        for paper in tqdm(papers, desc="CF gen over papers"):
            if self._cache and paper.id in self._cache:
                cf_paper = self._cache[paper.id][1]
                res[paper.id] = cf_paper
                continue

            try:
                cf_paper = self.single_run(paper, **config)
            except ValueError as e:
                logging.error(f"Could not generate counterfactual for paper {paper.id}: {e}")
                errors[paper.id] = str(e)
                continue

            res[paper.id] = cf_paper
            if self._cache is not None:
                self._cache[paper.id] = (paper, cf_paper)

            # cache intermediate result if cache is set
            if len(res) % self._disk_cache_freq == 0 and self._disk_cache_dir is not None:
                PaperCounterfactualDataset({pid: next(p for p in papers if p.id == pid) for pid in res}, res).save(self._disk_cache_dir, force=True)

        res = PaperCounterfactualDataset({paper.id: paper for paper in papers}, res, self.name)
        if res.meta is None:
            res.meta = {}
        res.meta["errors"] = errors

        # cache result
        if self._disk_cache_dir is not None:
            res.save(self._disk_cache_dir, force=True)

        # return final result
        return res
