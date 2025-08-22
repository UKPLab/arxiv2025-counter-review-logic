from ....data import Paper, Review
from ....framework import AutomaticReviewGenerator


class DebugARGtor(AutomaticReviewGenerator):
    """
    This class is intended for debugging purposes only. It generates a dummy review report.
    """
    def __init__(self, name):
        super().__init__(name)

    def run(self, paper: Paper, **config) -> Review:
        return Review(f"{self.name} This is a dummy review report. I think the o_paper lacks novelty"*3, {"overall": 1})