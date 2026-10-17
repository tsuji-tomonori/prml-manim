"""Independently enumerate the eight observed configurations and compare dynamic programs."""
import json
from itertools import product
from pathlib import Path
import numpy as np
from chain_model import TABLES, candidates, sum_messages, max_trace

rows = []
for state in product((0, 1), repeat=4):
    if state[3] != 1:
        continue
    w = 1
    for i in range(3):
        w *= int(TABLES[i, state[i], state[i + 1]])
    rows.append((state, w))
assert rows == candidates()
assert len(rows) == 8
total = sum(w for _, w in rows)
by_middle = [sum(w for s, w in rows if s[1] == value) for value in (0, 1)]
assert total == 219 and by_middle == [75, 144]
assert np.array_equal(sum_messages()[0], [5, 8])
assert np.array_equal(sum_messages()[1], [15, 18])
assert np.array_equal(sum_messages()[2], by_middle)
winner = max(rows, key=lambda row: row[1])
assert winner == ((1, 0, 0, 1), 48)
assert max_trace()[:2] == winner
assert sum(w for s, w in rows if s[1] == 1) / total > .65
assert winner[0][1] == 0
result = dict(tables=TABLES.tolist(), observed='x4=1', candidates=[dict(state=list(s), weight=w) for s,w in rows],
              normalizer=total, x2_weights=by_middle, x2_one_probability=144/219,
              left_message=sum_messages()[0].tolist(), right_message=sum_messages()[1].tolist(),
              map_state=list(winner[0]), map_weight=winner[1], map_probability=48/219)
(Path(__file__).with_name('numerical_validation.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS', {k:v for k,v in result.items() if k not in ('tables','candidates')})
