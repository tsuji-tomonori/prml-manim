"""Independent checks of the displayed evidence, sparsity, posterior and prediction."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from rvm_model import X,T,PHI,WIDTH,BETA,MODEL,basis,evidence,one_basis_terms,optimum_alpha,predict
root=Path(__file__).resolve().parent
active=MODEL['active'].tolist()
assert active==[4,8],active
assert np.all(np.diff([v['evidence'] for v in MODEL['history']])>=-1e-9)
assert np.allclose(PHI,basis(X))
assert np.max(np.abs(np.diag(MODEL['sigma'])-np.diag(MODEL['sigma'].T)))<1e-12
# Equation (7.97): the one-basis change in log evidence must equal the direct difference.
checks=[]
for i in range(PHI.shape[1]):
    s,q=one_basis_terms(MODEL['alpha'],i)
    trial=MODEL['alpha'].copy();trial[i]=np.inf
    before=evidence(trial)
    chosen=optimum_alpha(s,q)
    if np.isfinite(chosen):
        trial[i]=chosen
        direct=evidence(trial)-before
        formula=.5*(np.log(chosen)-np.log(chosen+s)+q*q/(chosen+s))
        assert abs(direct-formula)<1e-10
    checks.append(dict(index=i,center=(None if i==0 else float(X[i-1])),s=s,q=q,criterion=q*q-s,selected=bool(np.isfinite(MODEL['alpha'][i]))))
# Independently calculate the Gaussian predictive variance from posterior sampling.
rng=np.random.default_rng(72)
weights=rng.multivariate_normal(MODEL['mean'],MODEL['sigma'],120000)
z=.45;phi=basis([z])[:,MODEL['active']][0]
samples=weights@phi+rng.normal(0,1/np.sqrt(BETA),len(weights))
mu,var,wvar=predict(z)
assert abs(mu-float(samples.mean()))<.003
assert abs(var-float(samples.var()))<.001
assert abs(var-(1/BETA+wvar))<1e-12
rmse=float(np.sqrt(np.mean((predict(X)[0]-T)**2)))
assert rmse<.03
outer=predict(1.4)
assert outer[1]<var # This local-kernel example becomes falsely confident away from data.
result=dict(active_indices=active,relevance_centers=X[np.array(active)-1].tolist(),alpha=MODEL['alpha'][MODEL['active']].tolist(),posterior_mean=MODEL['mean'].tolist(),posterior_covariance=MODEL['sigma'].tolist(),training_rmse=rmse,prediction_x=z,prediction_mean=mu,prediction_variance=var,weight_variance=wvar,noise_variance=1/BETA,interval_two_sd=[mu-2*np.sqrt(var),mu+2*np.sqrt(var)],outside_x=1.4,outside_prediction=dict(mean=outer[0],variance=outer[1]),history=MODEL['history'],candidates=checks,sampling=dict(n=len(samples),mean=float(samples.mean()),variance=float(samples.var())))
(root/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS', {k:v for k,v in result.items() if k not in ['posterior_covariance','history','candidates']})
