from ...framework.rcd import ReviewChangeDetector, ReviewDelta


class MultiReviewChangeDetector(ReviewChangeDetector):
    def __init__(self, rcds: list[ReviewChangeDetector]):
        super().__init__("multi_rcd_" + ".".join([rcd.name for rcd in rcds]))
        self.rcds = rcds

    def run(self, review1, review2, **config) -> ReviewDelta:
        result = {
        }
        for rcd in self.rcds:
            result[rcd.name] = rcd.run(review1, review2)

        return ReviewDelta(
            review1,
            review2,
            change_desc={
                rcd_name: delta.change for rcd_name, delta in result.items()
            }
        )