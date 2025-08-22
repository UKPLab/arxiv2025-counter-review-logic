import logging
from collections import Counter

import numpy as np

from cerg.framework.eval import ReviewDeltaEvaluator
from cerg.framework.rcd import ReviewDelta


class PointRDE(ReviewDeltaEvaluator):
    def __init__(self):
        super().__init__("PointRDE")

    def single_delta_stats(self, delta: ReviewDelta):
        stats = {}
        change = delta.change

        if "shared_points" not in change or "contradictory_points" not in change:
            logging.warning("Change does not contain 'shared_points' or 'contradictory_points'.", change)
            return None

        shared = len(change["shared_points"])
        contra = len(change["contradictory_points"])

        points_r1 = set(m["id"] for m in change["points_review1"])
        points_r2 = set(m["id"] for m in change["points_review2"])

        cnt = 0
        new_ids = {}
        for s in change["shared_points"]:
            if s["id1"] in points_r1 or s["id2"] in points_r2:
                if s["id1"] in new_ids:
                    new_id = new_ids[s["id1"]]

                if s["id2"] in new_ids:
                    new_id = new_ids[s["id2"]]

                if s["id1"] in points_r1 and s["id2"] in points_r2:
                    new_id = f"shared_{cnt}"

                    new_ids[s["id1"]] = new_id
                    new_ids[s["id2"]] = new_id
                    cnt += 1

                if s["id1"] in points_r1:
                    points_r1.remove(s["id1"])

                if s["id2"] in points_r2:
                    points_r2.remove(s["id2"])

                points_r1.add(new_id)
                points_r2.add(new_id)

        if len(points_r1) + len(points_r2) - len(change["shared_points"]) == 0:
            jaccard = np.nan
        else:
            jaccard = len(points_r1.intersection(points_r2)) / (
                len(points_r1) + len(points_r2) - len(points_r1.intersection(points_r2))
            )

        stats["shared"] = shared / (len(points_r1) + len(points_r2)) if (len(points_r1) + len(points_r2)) > 0 else np.nan
        stats["contradictory"] = contra / (len(points_r1) + len(points_r2)) if (len(points_r1) + len(points_r2)) > 0 else np.nan
        stats["jaccard"] = jaccard

        # density of sentiments
        pos_density_r1 = len([m["sentiment"] for m in change["points_review1"] if "sentiment" in m and m["sentiment"] == "positive"]) / len(change["points_review1"]) if len(change["points_review1"]) > 0 else 0
        pos_density_r2 = len([m["sentiment"] for m in change["points_review2"] if "sentiment" in m and m["sentiment"] == "positive"]) / len(change["points_review2"]) if len(change["points_review2"]) > 0 else 0

        stats["positive_sentiment_density_diff"] = pos_density_r2 - pos_density_r1

        return stats

    def run(self, rd: ReviewDelta | list[ReviewDelta], **config) -> dict:
        stats = []
        if type(rd) != list:
            rd = [rd]

        for r in rd:
            stats += [self.single_delta_stats(r)]

        stats = [s for s in stats if s is not None] # exclude nones

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
