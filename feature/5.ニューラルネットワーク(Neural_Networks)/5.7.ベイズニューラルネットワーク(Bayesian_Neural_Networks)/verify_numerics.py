"""Independent finite differences, quadrature, projection and symmetry checks."""
import json
import numpy as np
from pathlib import Path
from scipy.integrate import quad
from bayesian_model import *

def main():
    d=np.load(ROOT/'model_data.npz');w=d['w'];a=d['A'];cov=d['cov']
    eps=1e-5;eye=np.eye(len(w))*eps
    numerical=np.column_stack([(output(w+h,X)-output(w-h,X))/(2*eps) for h in eye])
    jac_error=float(np.max(abs(numerical-jacobian(w,X))))
    assert jac_error<1e-7
    # Independent second directional derivative of scalar objective.
    rng=np.random.default_rng(570);v=rng.normal(size=len(w));v/=np.linalg.norm(v);h=2e-4
    fd=(objective(w+h*v,X,T,ALPHA,BETA)[0]-2*objective(w,X,T,ALPHA,BETA)[0]+objective(w-h*v,X,T,ALPHA,BETA)[0])/h**2
    hessian_error=float(abs(fd-v@a@v));assert hessian_error<1e-3
    assert np.linalg.eigvalsh(a).min()>0
    assert np.max(abs(a@cov-np.eye(7)))<1e-9
    # Gaussian projection variance, independent samples and analytic contraction.
    x=np.array([-2.4,0,2.4]);g=jacobian(w,x);pred=variance(w,cov,x)
    samples=rng.multivariate_normal(np.zeros(7),cov,120000)
    measured=(samples@g.T).var(axis=0)
    projection_error=float(np.max(abs(measured/pred-1)));assert projection_error<.02
    integral_errors=[]
    for sd in [.25,1.1,2.8]:
        q=quad(lambda z:gaussian(1.2,z,.35**2)*gaussian(z,0,sd**2),-np.inf,np.inf)[0]
        integral_errors.append(abs(q-evidence_example(sd)))
    assert max(integral_errors)<1e-9
    # Verify actual hyperparameter updates, not the illustrative four-direction bars.
    update_errors=[]
    for i in range(4):
        wi=d['history_w'][i];ga=d['history_gamma'][i]
        expected=[ga/(wi@wi),(len(T)-ga)/np.sum((output(wi,X)-T)**2)]
        update_errors.extend(abs(np.array(expected)-[d['history_alpha'][i+1],d['history_beta'][i+1]]))
    assert max(update_errors)<1e-9
    # Hidden-unit permutation gives exactly the same function.
    u,b,v,c=unpack(w,1);perm=np.r_[u[::-1].ravel(),b[::-1],v[::-1],c]
    symmetry_error=float(np.max(abs(output(w,GRID)-output(perm,GRID))));assert symmetry_error<1e-12
    cw=d['cw'];cx=d['cx'];ca=d['cA'];cc=d['ccov']
    assert np.linalg.eigvalsh(ca).min()>0
    activation=output(cw,cx);variances=variance(cw,cc,cx)
    plain=expit(activation);bayes=expit(kappa(variances)*activation)
    assert np.all(abs(bayes-.5)<=abs(plain-.5)+1e-12)
    assert np.all((bayes>.5)==(plain>.5))
    actual=integrated_sigmoid(2,9)
    independent=quad(lambda z:expit(z)*gaussian(z,2,9),-40,40,epsabs=1e-12)[0]
    assert abs(actual-independent)<1e-7
    # Demonstrate why Eq 5.190 needs a_MAP, not b^T w_MAP.
    wrong=jacobian(cw,cx)@cw
    correction_difference=float(np.max(abs(wrong-activation)));assert correction_difference>.1
    mean=output(w,GRID);vv=variance(w,cov,GRID)
    result=dict(jacobian_max_error=jac_error,hessian_direction_error=hessian_error,
        minimum_regression_curvature=float(np.linalg.eigvalsh(a).min()),map_gradient_norm=float(np.linalg.norm(objective(w,X,T,ALPHA,BETA)[1])),
        projection_relative_error=projection_error,evidence_quadrature_error=max(integral_errors),
        hyperparameter_update_error=float(max(update_errors)),permutation_error=symmetry_error,
        class_alpha=float(d['calpha']),class_log_evidence=d['class_evidence'].tolist(),
        minimum_classification_curvature=float(np.linalg.eigvalsh(ca).min()),class_gradient_max=float(d['class_grad'].max()),
        sigmoid_integral=actual,sigmoid_quadrature_difference=abs(actual-independent),sigmoid_kappa=float(expit(2*kappa(9))),
        sigmoid_kappa_error=float(abs(expit(2*kappa(9))-actual)),erratum_5190_difference=correction_difference,
        regression_variance_at_x=dict(zip(map(str,x),pred.tolist())),
        band_min=float((mean-2*np.sqrt(vv+1/BETA)).min()),band_max=float((mean+2*np.sqrt(vv+1/BETA)).max()),
        alpha_updates=d['history_alpha'].tolist(),beta_updates=d['history_beta'].tolist(),gamma_updates=d['history_gamma'].tolist())
    (ROOT/'numerical_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
