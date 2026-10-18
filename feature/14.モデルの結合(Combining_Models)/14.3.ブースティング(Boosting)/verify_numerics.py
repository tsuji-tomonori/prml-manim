"""Check every displayed AdaBoost number against exhaustive stump fitting."""
import json
from pathlib import Path
import numpy as np
from boosting_model import X,T,QUERY_INDEX,rounds
s=rounds()
assert [r['k'] for r in s]==[2,6,4]
assert [r['left'] for r in s]==[-1,-1,1]
assert np.allclose([r['epsilon'] for r in s],[.2,.125,3/14])
assert np.allclose([r['alpha'] for r in s],[np.log(4),np.log(7),np.log(11/3)])
assert np.allclose(np.array(s[0]['weight_after'])[[4,5]],[.25,.25])
assert np.allclose(np.array(s[0]['weight_after'])[[0,1,2,3,6,7,8,9]],[.0625]*8)
assert np.allclose(np.array(s[1]['weight_after'])[[2,3]],[.25,.25])
for r in s:
    pred=np.array(r['prediction']);before=np.array(r['weight_before']);after=np.array(r['weight_unnormalized'])
    assert np.allclose(after[pred==T],before[pred==T])
    assert np.allclose(after[pred!=T],before[pred!=T]*np.exp(r['alpha']))
    assert np.isclose(sum(r['weight_after']),1)
assert [s[0]['prediction'][QUERY_INDEX],s[1]['prediction'][QUERY_INDEX],s[2]['prediction'][QUERY_INDEX]]==[1,-1,-1]
assert np.isclose(s[-1]['score'][QUERY_INDEX],np.log(4)-np.log(7)-np.log(11/3))
assert s[0]['committee'][QUERY_INDEX]==1 and s[-1]['committee'][QUERY_INDEX]==-1
assert s[-1]['committee']==T.tolist()
result=dict(dataset=dict(x=X.tolist(),targets=T.tolist(),query_x=float(X[QUERY_INDEX]),query_target=int(T[QUERY_INDEX])),rounds=s,query_final_score=s[-1]['score'][QUERY_INDEX],final_training_correct=10)
Path('numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS', [r['epsilon'] for r in s], [r['alpha'] for r in s],result['query_final_score'])
