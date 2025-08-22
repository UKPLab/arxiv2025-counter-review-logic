from collections import Counter

import numpy as np

from cerg.framework.eval import ReviewDeltaEvaluator
from cerg.framework.rcd import ReviewDelta


class AspectRDE(ReviewDeltaEvaluator):
    def __init__(self):
        super().__init__("AspectRDE")

    def single_delta_stats(self, delta: ReviewDelta):
        stats = {}
        change = delta.change

        aspect_changes = change.get("aspect_changes", None)
        if aspect_changes is None:
            return None

        aspects_r1 = set(l for m in change["aspects"]["review1"] for l in m)
        aspects_r2 = set(l for m in change["aspects"]["review2"] for l in m)

        aspect_dist_r1 = Counter(l for m in change["aspects"]["review1"] for l in m)
        num_aspects_r1 = sum(aspect_dist_r1.values())
        aspect_dist_r2 = Counter(l for m in change["aspects"]["review2"] for l in m)
        num_aspects_r2 = sum(aspect_dist_r2.values())

        rl_related_aspects = [
            #"Accuracy",
            #"Analysis",
            "Contribution",
            "Discussion",
            #"Evaluation",
            "Evidence",
            "Findings",
            #"Improvement",
            "Interpretation",
            #"Performance",
            "Result",
            "Statistical Significance"
        ]

        rl_density_r1 = sum(aspect_dist_r1[a] for a in rl_related_aspects) / num_aspects_r1 if num_aspects_r1 > 0 else 0
        rl_density_r2 = sum(aspect_dist_r2[a] for a in rl_related_aspects) / num_aspects_r2 if num_aspects_r2 > 0 else 0

        div =  len(aspects_r1) + len(aspects_r2) - len(aspect_changes["common_aspects"])
        stats["jaccard"] = len(aspect_changes["common_aspects"]) / div if div > 0 else 0

        stats["num_added"] = len(aspect_changes["added_aspects"])
        stats["num_removed"] = len(aspect_changes["removed_aspects"])
        stats["density_soundness_diff"] = rl_density_r2 - rl_density_r1
        stats["soundness_diff"] = sum(aspect_dist_r2[a] for a in rl_related_aspects) - sum(aspect_dist_r1[a] for a in rl_related_aspects)

        return stats

    def run(self, rd: ReviewDelta | list[ReviewDelta], **config) -> dict:
        stats = []
        if type(rd) != list:
            rd = [rd]

        for r in rd:
            stats += [self.single_delta_stats(r)]

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
