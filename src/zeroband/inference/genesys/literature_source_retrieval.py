from typing import Dict

from zeroband.inference.genesys.literature_source_retrieval_utils import (
    weighted_literature_source_reward,
    extract_literature_info,
    calc_literature_similarity,
    calc_genere_similarity,
    calc_writer_similarity,
    calc_chapter_similarity,
    calc_verse_similarity,
)


#  TODO(INF800)
def compute_literature_source_retrieval_reward(
    completion: str, verification_info: Dict
):
    # Actual
    ground_truths = verification_info["ground_truth"]
    if not ground_truths:
        return 0

    literature_name_actual = ground_truths["literature_name"]
    genere_actual = ground_truths["genere"]
    writer_actual = ground_truths["writer"]
    chapter_actual = ground_truths["chapter"]
    verse_actual = ground_truths["verse"]

    # Predicted
    model_response = completion
    if "</think>" in model_response:
        model_analysis = model_response.split("</think>")[1]
    else:
        return 0

    model_answer = extract_literature_info(model_analysis)
    if not model_answer:
        return 0

    literature_name_predicted = model_answer["literature_name"]
    genere_predicted = model_answer["genere"]
    writer_predicted = model_answer["writer"]
    chapter_predicted = model_answer["chapter"]
    verse_predicted = model_answer["verse"]

    # Similarity scores
    lit_sim = calc_literature_similarity(
        literature_name_actual, literature_name_predicted
    )
    gnr_sim = calc_genere_similarity(genere_actual, genere_predicted)
    wrt_sim = calc_writer_similarity(writer_actual, writer_predicted)
    chp_sim = calc_chapter_similarity(chapter_actual, chapter_predicted)
    vrs_sim = calc_verse_similarity(verse_actual, verse_predicted)

    # Weighted scoring
    weights = [0.6, 0.1, 0.15, 0.1, 0.5]
    reward = weighted_literature_source_reward(
        [lit_sim, gnr_sim, wrt_sim, chp_sim, vrs_sim],
        weights,
    )

    return reward
