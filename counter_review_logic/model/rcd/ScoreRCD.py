from ...data import Review
from ...framework.rcd import ReviewChangeDetector, ReviewDelta
from .utils import get_review_score_data


class ScoreRCD(ReviewChangeDetector):
    """
    This class determines changes between two reviews considering the polarity.
    """
    def __init__(self):
        super().__init__("score_rcd")

    def run(self, review1: Review, review2: Review, **config) -> ReviewDelta:
        result = {}

        def score_delta(s1, s2):
            try:
                return s2 - s1
            except:
                return None

        data = get_review_score_data(review1, review2)
        r1_scores, r2_scores, common_scores = data["r1"], data["r2"], data["common_scores"]
        r1_overall, r2_overall = data["overall_score1"], data["overall_score2"]

        result["overall"] = r2_overall - r1_overall if r1_overall is not None and r2_overall is not None else None
        result["other_scores"] = {
            score:  score_delta(r1_scores[score], r2_scores[score]) for score in common_scores
        }

        return ReviewDelta(review1, review2, result)