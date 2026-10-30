from __future__ import annotations
from evaluation.citation_metrics import citation_faithfulness_from_records
from evaluation.retrieval_metrics import evidence_recall, query_hit_rate

def aggregate_records(records: list[dict], ks: list[int]) -> dict:
    all_claims=[claim for row in records for claim in row.get("claims",[])]
    judged=[c for c in all_claims if c.get("verdict") in {"SUPPORTED","UNSUPPORTED","CONTRADICTED"}]
    counts={v:sum(c.get("verdict")==v for c in judged) for v in ("SUPPORTED","UNSUPPORTED","CONTRADICTED")}
    n=len(judged); bad=counts["UNSUPPORTED"]+counts["CONTRADICTED"]
    retrieval={}
    for k in ks:
        recalls=[]; hits=[]
        for row in records:
            if len(row.get("retrieved_ids",[])) < k: continue
            value=evidence_recall(row.get("retrieved_ids",[]),row.get("gold_ids",[]),k)
            hit=query_hit_rate(row.get("retrieved_ids",[]),row.get("gold_ids",[]),k)
            if value is not None: recalls.append(value)
            if hit is not None: hits.append(hit)
        retrieval[str(k)]={"requested_k":k,"eligible_queries":len(recalls),"evidence_set_recall":sum(recalls)/len(recalls) if recalls else None,"query_level_hit_rate":sum(hits)/len(hits) if hits else None}
    citation=citation_faithfulness_from_records(all_claims)
    return {"examples":len(records),"failed_examples":sum(r.get("status")=="error" for r in records),"retrieval":retrieval,
        "claims":{"total":len(all_claims),"evaluable":n,"unevaluated":len(all_claims)-n,"supported":counts["SUPPORTED"],"unsupported":counts["UNSUPPORTED"],"contradicted":counts["CONTRADICTED"],"hallucination_rate":bad/n if n else None,"unsupported_rate":counts["UNSUPPORTED"]/n if n else None,"contradicted_rate":counts["CONTRADICTED"]/n if n else None},
        "citation_faithfulness":citation["citation_faithfulness"],"cited_claims":citation["cited_claims"],"faithful_cited_claims":citation["faithful_cited_claims"],
        "citation_metric_definition":"cited claims judged SUPPORTED with at least one cited chunk in supporting_evidence_ids, divided by all cited claims; automated critic proxy, requires human validation."}
