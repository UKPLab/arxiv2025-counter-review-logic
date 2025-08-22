from cerg.data import Review


def get_review_text_data(review1:Review, review2:Review):
    # review texts
    r1_full = review1.get_text()
    r2_full = review2.get_text()

    r1_by_section = review1.sections
    r2_by_section = review2.sections

    # 0) check if sections were added or removed
    added_sections = set(r2_by_section.keys()) - set(r1_by_section.keys())
    removed_sections = set(r1_by_section.keys()) - set(r2_by_section.keys())
    common_sections = set(r1_by_section.keys()) & set(r2_by_section.keys())

    return {
        "r1": r1_full,
        "r2": r2_full,
        "r1_sections": r1_by_section,
        "r2_sections": r2_by_section,
        "added_sections": added_sections,
        "removed_sections": removed_sections,
        "common_sections": common_sections
    }


def get_review_score_data(review1: Review, review2: Review):
    scores1 = review1.scores
    scores2 = review2.scores

    common_scores = set(scores1.keys()) & set(scores2.keys())

    overall_score1 = review1.get_overall_score()
    overall_score2 = review2.get_overall_score()

    return {
        "r1": scores1,
        "r2": scores2,
        "common_scores": common_scores,
        "overall_score1": overall_score1,
        "overall_score2": overall_score2
    }