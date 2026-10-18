"""Cross-check forward/backward, EM, and Viterbi against direct enumeration."""
import json
from pathlib import Path
import numpy as np
from hmm_model import PI, A, B, OBS, PATH_OBS, infer, em_step, viterbi, exhaustive


def main():
    assert np.allclose(A.sum(axis=1), 1) and np.allclose(B.sum(axis=1), 1)
    data = infer(OBS)
    paths = exhaustive(OBS)
    likelihood = sum(p for p, _ in paths)
    assert abs(data['likelihood']-likelihood) < 1e-14
    for t in range(len(OBS)):
        direct = sum(p for p,z in paths if z[t] == 1) / likelihood
        assert abs(data['gamma'][t,1]-direct)<1e-14
        first = exhaustive(OBS[:t+1])
        direct_filter = sum(p for p,z in first if z[t] == 1) / sum(p for p,z in first)
        assert abs(data['filter'][t,1]-direct_filter)<1e-14
    for t in range(len(OBS)-1):
        direct = np.array([[sum(p for p,z in paths if z[t]==j and z[t+1]==k)
                            for k in range(2)] for j in range(2)]) / likelihood
        assert np.allclose(data['xi'][t],direct,atol=1e-14)
    assert abs(data['filter'][1,1]-.5842696629213483)<1e-12
    assert abs(data['gamma'][1,1]-.18310980103168756)<1e-12
    update=em_step(OBS)
    assert update['new_likelihood'] >= update['old_likelihood']-1e-14
    assert np.allclose(update['a'].sum(axis=1),1)
    assert np.allclose(update['b'].sum(axis=1),1)
    winner=viterbi(PATH_OBS)
    direct=max(exhaustive(PATH_OBS))
    assert winner['path']==direct[1] and abs(winner['probability']-direct[0])<1e-14
    marginal=infer(PATH_OBS)['gamma']
    local=tuple(np.argmax(marginal,axis=1))
    assert winner['path'] != local
    local_probability=dict((z,p) for p,z in exhaustive(PATH_OBS))[local]
    assert winner['probability'] > local_probability
    out=dict(observations=list(OBS), filter_day2=float(data['filter'][1,1]),
             smooth_day2=float(data['gamma'][1,1]), likelihood=likelihood,
             em_transition_counts=update['transition_counts'].tolist(),
             em_transition=update['a'].tolist(),em_emission=update['b'].tolist(),
             em_likelihood=update['new_likelihood'],path_observations=list(PATH_OBS),
             viterbi_path=list(winner['path']),viterbi_probability=winner['probability'],
             local_path_probability=local_probability,local_marginal_path=[int(v) for v in local])
    Path('numerical_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
