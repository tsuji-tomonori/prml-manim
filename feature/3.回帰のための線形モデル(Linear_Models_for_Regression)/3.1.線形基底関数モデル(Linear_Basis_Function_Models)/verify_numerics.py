"""Independent numerical identities, ranges, and audio/script consistency."""
import json
from pathlib import Path
import numpy as np
from basis_model import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST, valid_entry, OUTPUT_DIR


def main():
    checks={}
    # Normal-equation residual, orthogonal projection and intercept identity.
    residual=T-PHI@W_ML
    np.testing.assert_allclose(PHI.T@residual,0,atol=1e-11)
    np.testing.assert_allclose(W_ML[0],T.mean()-PHI[:,1:].mean(axis=0)@W_ML[1:],atol=1e-11)
    P=PHI@np.linalg.pinv(PHI)
    np.testing.assert_allclose(P,P.T,atol=1e-11)
    np.testing.assert_allclose(P@P,P,atol=1e-11)
    checks['least_squares']={'normal_residual':float(np.linalg.norm(PHI.T@residual)),
                              'noise_variance_ml':float(np.mean(residual**2))}
    # The opening and ending use the same independent observation. Compare
    # displayed roundings as well as full-precision predictions.
    assert not np.any(np.isclose(X, HOLDOUT_X))
    np.testing.assert_allclose(HOLDOUT_T, target(.68)+.06)
    np.testing.assert_allclose(HOLDOUT_LINEAR, np.array([HOLDOUT_X,1.])@LINEAR_COEF)
    np.testing.assert_allclose(HOLDOUT_RIDGE, design([HOLDOUT_X])@fit(.1),atol=1e-12)
    assert abs(HOLDOUT_T-HOLDOUT_RIDGE)<abs(HOLDOUT_T-HOLDOUT_LINEAR)
    assert (round(HOLDOUT_T,2),round(HOLDOUT_LINEAR,2),round(HOLDOUT_RIDGE,2))==(-.92,-.24,-.86)
    checks['heldout_story']={'x':HOLDOUT_X,'observation':HOLDOUT_T,
        'linear_prediction':HOLDOUT_LINEAR,'ridge_lambda':HOLDOUT_RIDGE_LAMBDA,
        'ridge_prediction':HOLDOUT_RIDGE,'linear_abs_error':abs(HOLDOUT_T-HOLDOUT_LINEAR),
        'ridge_abs_error':abs(HOLDOUT_T-HOLDOUT_RIDGE)}
    # Actual normalized Gaussian log likelihood, independently evaluated.
    beta=12.;w=fit(.1)
    logp=np.log(np.sqrt(beta/(2*np.pi))*np.exp(-beta*(T-PHI@w)**2/2)).sum()
    np.testing.assert_allclose(logp,len(X)/2*np.log(beta/(2*np.pi))-beta*error(w))
    checks['gaussian_likelihood']=True
    norms=[];maxcurve=0
    grid=design(np.linspace(0,1,2001))
    for lam in np.logspace(-6,2,161):
        w=fit(lam)
        np.testing.assert_allclose(PHI.T@(PHI@w-T)+lam*w,0,atol=1e-10)
        norms.append(np.linalg.norm(w));maxcurve=max(maxcurve,float(np.max(np.abs(grid@w))))
    assert np.all(np.diff(norms)<=1e-10)
    assert maxcurve<1.8
    checks['ridge']={'max_curve_abs':maxcurve,'norm_at_1e_minus6':norms[0],'norm_at_100':norms[-1]}
    # Verify one-point gradient using finite differences, not the same update code.
    i=ORDER[0];w=LMS[0];delta=1e-6;gradient=[]
    def cost(v):return .5*(T[i]-PHI[i]@v)**2
    for j in range(len(w)):
        h=np.eye(len(w))[j]*delta
        gradient.append((cost(w+h)-cost(w-h))/(2*delta))
    np.testing.assert_allclose(LMS[1],w-.22*np.array(gradient),atol=1e-9)
    assert T[ORDER[0]]>0 and T[ORDER[1]]-PHI[ORDER[1]]@LMS[1]<0
    assert np.max(np.abs(grid@LMS.T))<1.65
    overshoot=lms_step(LMS[2],ORDER[2],.45)
    assert PHI[ORDER[2]]@overshoot>T[ORDER[2]]
    assert np.max(np.abs(grid@overshoot))<1.65
    checks['lms']={'overshoot_prediction':float(PHI[ORDER[2]]@overshoot),'target':float(T[ORDER[2]])}
    # Dense boundary search independently confirms both constrained minima.
    for q in [1,1.1,1.25,1.5,1.75,2]:
        opt=constraint_solution(q)
        points=q_boundary(q,200001)
        costs=np.sum((points-CONSTRAINT_TARGET)**2,axis=1)
        np.testing.assert_allclose(np.min(costs),np.sum((opt-CONSTRAINT_TARGET)**2),atol=1e-8)
        np.testing.assert_allclose(np.sum(np.abs(points)**q,axis=1),1,atol=1e-12)
    checks['constraints']={'tested_q':[1,1.1,1.25,1.5,1.75,2],'l1':L1_POINT.tolist(),'l2':L2_POINT.tolist()}
    both=fit(t=np.column_stack([T,T2]))
    np.testing.assert_allclose(both[:,0],fit(t=T),atol=1e-10)
    np.testing.assert_allclose(both[:,1],fit(t=T2),atol=1e-10)
    shifted=fit(t=np.column_stack([T+.3,T2]))
    np.testing.assert_allclose(grid@shifted[:,0],grid@both[:,0]+.3,atol=1e-10)
    np.testing.assert_allclose(shifted[:,1],both[:,1],atol=1e-10)
    checks['multiple_outputs']=True
    # Independent finite-difference check of the one-point contour example.
    point=np.array([1.5,1.]); h=1e-6
    cost=lambda w:.5*(w.sum()-1)**2
    gradient=np.array([(cost(point+h*e)-cost(point-h*e))/(2*h) for e in np.eye(2)])
    np.testing.assert_allclose(gradient,[1.5,1.5],atol=1e-9)
    after=point-.2*gradient
    np.testing.assert_allclose([cost(point),cost(after)],[1.125,.405],atol=1e-9)
    example=np.array([[1,2],[1,1]])@np.array([3,-1])
    np.testing.assert_array_equal(example,[1,2])
    checks['visual_aids']={'row_predictions':example.tolist(),
                           'gradient':gradient.tolist(),'error_before':cost(point),'error_after':cost(after)}
    entries=json.loads(MANIFEST.read_text())['scenes']
    assert len(entries)==len(SCENES)==9
    for scene,e in zip(SCENES,entries):assert valid_entry(scene,e),scene['id']
    assert {p.stem for p in OUTPUT_DIR.glob('scene*.wav')}=={s['id'] for s in SCENES}
    checks['audio']={'scenes':len(entries),'sentences':sum(len(e['subtitle_cues']) for e in entries),'duration':sum(e['duration'] for e in entries)}
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    Path('numerical_results.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
