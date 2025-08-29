from collections import Counter

import numpy as np

from ...framework import ReviewDeltaEvaluator
from ...framework import ReviewDelta


class ScoreRDE(ReviewDeltaEvaluator):
    def __init__(self):
        super().__init__("ScoreRDE")

    def single_delta_stats(self, delta: ReviewDelta):
        stats = {}
        change = delta.change

        # find overall score
        if "other_scores" not in change:
            return None

        overall_score = next((k for k in change["other_scores"].keys() if "overall" in k.lower() or "Rating" == k), None)
        if overall_score is None:
            return None

        stats["overall"] = change["other_scores"][overall_score]
        for score in change.get("other_scores", {}):
            stats[f"score_{score}"] = change["other_scores"][score]

        return stats

    def run(self, rd: ReviewDelta | list[ReviewDelta], **config) -> dict:
        stats = []
        if type(rd) != list:
            rd = [rd]

        for r in rd:
            stats += [self.single_delta_stats(r)]

        stats = [s for s in stats if s is not None]  # exclude nones

        # aggregate stats
        if len(stats) == 0:
            return {}

        keys = list(stats[0].keys())
        aggregated = {k: [] for k in keys}
        for stat in stats:
            for k in keys:
                if k in stat and stat[k] is not None:
                    aggregated[k].append(stat[k])

        # compute mean and std for numeric values
        for k in aggregated.keys():
            if isinstance(aggregated[k], list):
                s = [i for i in aggregated[k] if not np.isnan(i) and i is not None]

                if len(s) > 0:
                    aggregated[k] = {
                        "mean": float(np.nanmean(s)),
                        "std": float(np.nanstd(s)),
                        "min": float(np.nanmin(s)),
                        "max": float(np.nanmax(s)),
                        "median": float(np.nanmedian(s)),
                        "count": len(s)
                    }
                else:
                    aggregated[k] = {
                        "mean": None,
                        "std": None,
                        "min": None,
                        "max": None,
                        "median": None,
                        "count": 0
                    }
            else:
                aggregated[k] = Counter(aggregated[k])

        aggregated["raw"] = dict(zip([f"{r.review1.id}#{r.review2.id}" for r in rd], stats))

        return aggregated
