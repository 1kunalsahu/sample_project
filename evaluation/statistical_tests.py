from __future__ import annotations
import math

def paired_binary(baseline: list[bool|None], critic: list[bool|None]) -> dict:
    if len(baseline)!=len(critic): raise ValueError("Paired arrays must have the same length")
    pairs=[(a,b) for a,b in zip(baseline,critic) if a is not None and b is not None]
    b_only=sum(a and not b for a,b in pairs); c_only=sum(not a and b for a,b in pairs); n=len(pairs)
    # Exact two-sided McNemar test via binomial tail, no dependency required.
    m=b_only+c_only
    if not m: p=1.0
    else:
        tail=sum(math.comb(m,i) for i in range(min(b_only,c_only)+1))/(2**m)
        p=min(1.0,2*tail)
    return {"paired_n":n,"baseline_only_positive":b_only,"critic_only_positive":c_only,"mcnemar_exact_p":p}
