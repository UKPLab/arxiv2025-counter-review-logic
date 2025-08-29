from ...framework import ReviewChangeDetector, ReviewDelta


class DebugReviewChangeDetector(ReviewChangeDetector):
    def __init__(self):
        super().__init__("debug_rcd")

    def run(self, review1, review2, **config) -> ReviewDelta:
        return ReviewDelta(review1, review2, {
            "changes": "none, this is a debug detector"
        })