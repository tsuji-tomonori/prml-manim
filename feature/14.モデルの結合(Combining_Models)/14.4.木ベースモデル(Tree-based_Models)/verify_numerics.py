"""Recompute every numerical statement in the video from the example data."""
import json
from pathlib import Path
import numpy as np
from tree_model import X, T, QUERY, NEXT_T, grow, predict, cost, sse

stages = grow()
assert len(X) == 16 and len(stages) == 5
assert [(round(s[1][0]), s[1][1]) for s in stages[1:]] == [(0, 2.), (1, 2.), (1, 3.), (0, 3.)]
assert [len(s[0]) for s in stages] == [1, 2, 3, 4, 5]
assert np.isclose(sum(x.error for x in stages[2][0]), 31.5)
assert np.isclose(sum(x.error for x in stages[-1][0]), 0)
assert np.isclose(predict(stages[-1][0], QUERY), 14)
assert np.isclose(predict(stages[2][0], QUERY), 8.75)
assert np.isclose(cost(stages[2][0], 18), 85.5)
assert np.isclose(cost(stages[-1][0], 18), 90)
assert cost(stages[2][0], 18) < min(cost(s[0], 18) for i,s in enumerate(stages) if i != 2)
assert np.isclose(abs(predict(stages[2][0], QUERY) - NEXT_T), .75)
assert np.isclose(sse(T), 153.75)
def impurity(p):
    portions = np.array([p, 1-p], dtype=float)
    positive = portions[portions > 0]
    return float(-np.sum(positive*np.log(positive))), float(np.sum(portions*(1-portions)))

assert np.allclose(impurity(0), (0, 0))
assert np.allclose(impurity(.5), (np.log(2), .5))
assert np.allclose(impurity(1), (0, 0))
result = dict(points=X.tolist(), targets=T.tolist(), query=QUERY.tolist(), next_target=NEXT_T,
              steps=[dict(leaves=len(leaves), split=split, residual=sum(leaf.error for leaf in leaves),
                          cost_lambda18=cost(leaves, 18), query_prediction=predict(leaves, QUERY))
                     for leaves, split in stages], pruned_prediction=predict(stages[2][0], QUERY))
result['classification_impurity'] = {str(p):dict(entropy=impurity(p)[0], gini=impurity(p)[1]) for p in (0, .5, 1)}
(Path(__file__).parent / 'numerical_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print('PASS', [(s['leaves'], s['residual'], s['cost_lambda18']) for s in result['steps']])
