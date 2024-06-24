from typing import Dict, List, Tuple

from pytrec_eval import RelevanceEvaluator


class BeirEvaluator:

    def __init__(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        ignore_identical_ids: bool = True,  # noqa: FBT001, FBT002
    ):
        self.qrels = qrels
        self.results = results
        self.k_values = k_values
        self.ignore_identical_ids = ignore_identical_ids

    def evaluate(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        if self.ignore_identical_ids:
            print(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set "
                "``ignore_identical_ids=False`` to ignore this."
            )
            popped = []
            for qid, rels in self.results.items():
                for pid in list(rels):
                    if qid == pid:
                        self.results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in self.k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in self.k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in self.k_values])
        recall_string = "recall." + ",".join([str(k) for k in self.k_values])
        precision_string = "P." + ",".join([str(k) for k in self.k_values])
        evaluator = RelevanceEvaluator(self.qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(self.results)

        for query_id in scores.keys():
            for k in self.k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        for k in self.k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        for eval_metric in [ndcg, _map, recall, precision]:
            print("\n")
            for k in eval_metric.keys():  # type: ignore
                print(f"{k}: {eval_metric[k]:.4f}")  # type: ignore

        return ndcg, _map, recall, precision
