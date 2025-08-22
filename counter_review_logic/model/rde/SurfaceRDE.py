from collections import Counter

import numpy as np

from cerg.framework.eval import ReviewDeltaEvaluator
from cerg.framework.rcd import ReviewDelta


class SurfaceRDE(ReviewDeltaEvaluator):
    def __init__(self):
        super().__init__("SurfaceRDE")

    def single_delta_stats(self, delta: ReviewDelta):
        stats = {}
        change = delta.change

        stats["number_edit_distance"] = len(change.get("levensthein_edits", []))
        stats["edit_distribution"] = Counter([e[0] for e in change.get("levensthein_edits", [])]).most_common()

        stats["common_vocab_overlap"] = len(set(e[0] for e in change["vocab_edits"].get("most_common1", [])) & \
                                            set(e[0] for e in change["vocab_edits"].get("most_common2", []))) / (
                                                    len(change["vocab_edits"].get("most_common1", [])) + len(
                                                change["vocab_edits"].get("most_common2", [])))
        stats["3gram_overlap"] = len(set(e[0] for e in change["3gram_overlap"].get("most_common_3grams1", [])) & \
                                     set(e[0] for e in change["3gram_overlap"].get("most_common_3grams2", []))) / (
                                             len(change["vocab_edits"].get("most_common1", [])) + len(
                                         change["vocab_edits"].get("most_common2", [])))

        stats["changed_sections"] = len(change.get("section_changes", {}).get("added", [])) + \
                                     len(change.get("section_changes", {}).get("removed", []))
        stats["sectionwise_token_count_changes"] = change["by_section"]["token_count_changes"]
        stats["sectionwise_edit_distance"] = {k: len(v) for k,v in change["by_section"]["levensthein_edits"].items()}

        stats["rouge-1"] = change.get("rouge-1", np.nan)
        stats["rouge-2"] = change.get("rouge-2", np.nan)
        stats["rouge-l"] = change.get("rouge-l", np.nan)

        stats["sectionwise_rouge-l"] = change["by_section"].get("rouge-l", {})

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
                if k in stat:
                    aggregated[k].append(stat[k])

        for stat in ["number_edit_distance",
                     "common_vocab_overlap",
                     "3gram_overlap",
                     "changed_sections",
                     "rouge-1",
                     "rouge-2",
                     "rouge-l"]:
            s = [i for i in aggregated[stat] if not np.isnan(i) and i is not None]

            if len(s) > 0:
                aggregated[stat] = {
                    "mean": float(np.nanmean(s)),
                    "std": float(np.nanstd(s)),
                    "min": float(np.nanmin(s)),
                    "max": float(np.nanmax(s)),
                    "median": float(np.nanmedian(s)),
                    "count": len(s)
                }
            else:
                aggregated[stat] = {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "median": None,
                    "count": 0
                }

        # edit distribution
        edit_dist = Counter()
        for e in aggregated["edit_distribution"]:
            for k in e:
                edit_dist[k[0]] += k[1]

        s = sum(v for k,v in edit_dist.most_common())
        aggregated["edit_distribution"] = {k: float(v)/s for k, v in edit_dist.most_common()}

        # sectionwise
        for stat in [
            "sectionwise_token_count_changes",
            "sectionwise_edit_distance",
            "sectionwise_rouge-l"
        ]:
            stat_per_section = aggregated[stat]
            agg_per_section = {}

            for d in stat_per_section:
                for section, rs in d.items():
                    agg_per_section[section] = agg_per_section.get(section, [])
                    agg_per_section[section].append(rs)

            aggregated[stat] = {}
            for section, values in agg_per_section.items():
                aggregated[stat][section] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "median": float(np.nanmedian(values))
                }

        aggregated["raw"] = dict(zip([f"{r.review1.id}#{r.review2.id}" for r in rd], stats))

        return aggregated
