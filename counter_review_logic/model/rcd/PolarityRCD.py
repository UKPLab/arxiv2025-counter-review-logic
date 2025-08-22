import spacy

from cerg.data import Review
from cerg.framework.rcd import ReviewChangeDetector, ReviewDelta
from cerg.models.rcds.utils import get_review_text_data

import Levenshtein


class PolarityRCD(ReviewChangeDetector):
    """
    This class determines changes between two reviews considering the polarity.
    """
    def __init__(self):
        super().__init__("polarity_rcd")

    def run(self, review1: Review, review2: Review, **config) -> ReviewDelta:
        result = {"by_section": {}}

        data = get_review_text_data(review1, review2)
        r1_full, r2_full = data["r1"], data["r2"]

        r1_by_section, r2_by_section = data["r1_sections"], data["r2_sections"]
        common_sections = data["common_sections"]

        # TODO implement

        return ReviewDelta(review1, review2, result)