import random
import re

from cerg.data import Paper
from cerg.framework.cfg import PaperCounterfactualGenerator, PaperCounterfactual


class PaperLayoutCF(PaperCounterfactualGenerator):
    """
    This counterfactual generator is meant mostly for debugging. We simply return the same paper but
    with slightly altered layout -- deterministic.
    """
    def __init__(self, name=None):
        super().__init__(name="paper_layout" if name is None else name)

    def single_run(self, paper:Paper, **config) -> PaperCounterfactual:
        paper = paper.without_appendix()
        md = paper.md[:]

        # find all figures and move them to the end of the paper
        figures = [f["text"] for f in paper.get_figures().values()]
        for f in figures:
            md = md.replace(f, "")

        # find all tables and move them to the end of the paper
        tables = [t for t in paper.get_tables().values()]
        for t in tables:
            md = md.replace(t, "")

        # find all algorithms and move them to the end of the paper
        algorithms = [t for t in paper.get_algorithms().values()]
        for a in algorithms:
            md = md.replace(a, "")

        md += "\n\n## Figures and Tables\n"

        for m in figures + tables + algorithms:
            md += "\n" + m + "\n"

        # extend white spaces
        md = md.replace("\n", "\n" + "\n" * random.randint(0, 2))

        cf = Paper(
            id = paper.id,
            meta = paper.meta,
            md=md
        )

        return PaperCounterfactual(paper, self.name, cf, ["replaced \\n by random number of \\n", "moved all figures and tables to the very back of the paper"])