import logging
from collections import Counter

import numpy as np

from ...framework import ReviewDeltaEvaluator
from ...framework import ReviewDelta

from fuzzysearch import find_near_matches

class SoundnessRDE(ReviewDeltaEvaluator):
    def __init__(self):
        super().__init__("SoundnessRDE")

    def single_delta_stats(self, delta: ReviewDelta):
        stats = {}
        change = delta.change
        sub_deltas = list(change.keys())

        # get subdeltas
        point_delta = change["point"]
        aspect_delta = change["aspect"]

        if point_delta is None or aspect_delta is None:
            logging.warning("Change does not contain point or aspect delta.", change)
            return None

        if "shared_points" not in point_delta or "contradictory_points" not in point_delta:
            logging.warning("Change does not contain 'shared_points' or 'contradictory_points'.", point_delta)
            return None

        points_r1 = set(m["id"] for m in point_delta["points_review1"])
        points_r2 = set(m["id"] for m in point_delta["points_review2"])

        cnt = 0
        new_ids = {}
        for s in point_delta["shared_points"]:
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

        added_points = points_r2.difference(points_r1)
        removed_points = points_r1.difference(points_r2)

        # get aspects of added and removed points
        added_aspects = {}
        for p2id in added_points:
            assert not p2id.startswith("shared")

            point_desc = next(p for p in point_delta["points_review2"] if p["id"] == p2id)

            added_aspects[p2id] = set()

            for sentence, aspects in zip(aspect_delta["sentences"]["review2"], aspect_delta["aspects"]["review2"]):
                if len(sentence) < 5:
                    continue

                matches = find_near_matches(sentence.lower().strip(), point_desc["span"].lower().strip(), max_l_dist=10)
                if len(matches) > 0:
                    added_aspects[p2id].update(aspects)
                    break

        removed_aspects = {}
        for p1id in removed_points:
            assert not p1id.startswith("shared")

            point_desc = next(p for p in point_delta["points_review1"] if p["id"] == p1id)

            removed_aspects[p1id] = set()

            for sentence, aspects in zip(aspect_delta["sentences"]["review1"], aspect_delta["aspects"]["review1"]):
                if len(sentence) < 5:
                    continue

                matches = find_near_matches(sentence.lower().strip(), point_desc["span"].lower().strip(), max_l_dist=10)
                if len(matches) > 0:
                    removed_aspects[p1id].update(aspects)

        # soundness-related aspects
        rl_related_aspects = {
            "Accuracy",
            "Analysis",
            "Contribution",
            "Discussion",
            "Evaluation",
            "Evidence",
            "Findings",
            "Improvement",
            "Interpretation",
            "Performance",
            "Result",
            "Statistical Significance"
        }

        # get number of added/removed soundness points
        added_soundness_points = 0
        removed_soundness_points = 0
        for pid, aspects in added_aspects.items():
            if rl_related_aspects.intersection(aspects):
                added_soundness_points += 1

        for aspects in removed_aspects.values():
            if rl_related_aspects.intersection(aspects):
                removed_soundness_points += 1

        soundness_sentiments_r1 = []
        for aspect, sent in zip(aspect_delta["aspects"]["review1"], aspect_delta["sentences"]["review1"]):
            if len(sent) < 5 or len(aspect) == 0:
                continue

            if len(set(rl_related_aspects).intersection(set(aspect))) == 0:
                continue

            for point in point_delta["points_review1"] :
                span = point["span"].lower().strip()
                matches = find_near_matches(sent.lower().strip(), span, max_l_dist=10)
                if len(matches) > 0:
                    soundness_sentiments_r1 += [point["sentiment"]]

        soundness_sentiments_r2 = []
        for aspect, sent in zip(aspect_delta["aspects"]["review2"], aspect_delta["sentences"]["review2"]):
            if len(sent) < 5 or len(aspect) == 0:
                continue

            if len(set(rl_related_aspects).intersection(set(aspect))) == 0:
                continue

            for point in point_delta["points_review2"] :
                span = point["span"].lower().strip()
                matches = find_near_matches(sent.lower().strip(), span, max_l_dist=10)
                if len(matches) > 0:
                    soundness_sentiments_r2 += [point["sentiment"]]

        stats["added_soundness_points"] = added_soundness_points
        stats["removed_soundness_points"] = removed_soundness_points
        stats["soundess_points_diff"] = added_soundness_points - removed_soundness_points

        stats["soundness_sentiment_diff"] = Counter(soundness_sentiments_r1)["positive"] / len(soundness_sentiments_r1) - Counter(soundness_sentiments_r2)["positive"] / len(soundness_sentiments_r2) if len(soundness_sentiments_r1) > 0 and len(soundness_sentiments_r2) > 0 else 0

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
