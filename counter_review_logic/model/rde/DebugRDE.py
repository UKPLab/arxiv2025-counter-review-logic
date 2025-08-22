from cerg.framework.eval import ReviewDeltaEvaluator
from cerg.framework.rcd import ReviewDelta


class DebugReviewDeltaEvaluator(ReviewDeltaEvaluator):
    def __init__(self):
        super().__init__("ReviewDeltaEvaluator for Debugging")

    def run(self, rd: ReviewDelta|list[ReviewDelta], **config) -> dict:
        return {
            "score": 1.0,
            "note": "this evaluator is only for debugging"
        }