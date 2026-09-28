"""Independent finite differences and numerical identities for the video."""
import json
from pathlib import Path
import numpy as np
import regularization_model as m

def finite_gradient(fun,x,h=1e-5):
    eye=np.eye(len(x))*h
    return np.array([(fun(x+e)-fun(x-e))/(2*h) for e in eye])

def main():
    results={}
    w=m.initial(3,44);loss,g=m.objective(w,.04)
    num=finite_gradient(lambda v:m.objective(v,.04)[0],w)
    err=float(np.max(abs(num-g)));assert err<1e-7
    results['network_gradient_max_error']=err
    logs,ws=m.ridge_path();i=int(np.argmin([m.mse(w,m.XV,m.TV) for w in ws]))
    assert np.linalg.norm(ws[-1])<np.linalg.norm(ws[0])
    assert m.mse(ws[i],m.XV,m.TV)<m.mse(ws[0],m.XV,m.TV)
    results['ridge']={'best_lambda':float(10**logs[i]),'validation_mse':m.mse(ws[i],m.XV,m.TV),'weak_validation_mse':m.mse(ws[0],m.XV,m.TV),'norm_weak':float(np.linalg.norm(ws[0])),'norm_strong':float(np.linalg.norm(ws[-1]))}
    # Full affine change of units on both input and output.
    w=m.fit(3);a,b,v,c=m.unpack(w);A,B,C,D=2.3,.7,1.4,-.6
    transformed=np.r_[a/A,b-a*B/A,C*v,C*c+D]
    err=float(np.max(abs(m.predict(transformed,A*m.GRID+B)-(C*m.predict(w,m.GRID)+D))))
    assert err<1e-12
    pen=.4*(a@a)+.7*(v@v)
    pen_trans=.4*A**2*((a/A)@(a/A))+.7/C**2*((C*v)@(C*v))
    assert abs(pen-pen_trans)<1e-12
    results['affine_invariance_max_error']=err
    weights,tr,va=m.training_history();best=int(np.argmin(va))
    assert 0<best<len(va)-1 and va[best]<va[-1] and tr[-1]<tr[best]
    results['early_stop']={'step':best*20,'best_validation_mse':float(va[best]),'final_validation_mse':float(va[-1]),'final_training_mse':float(tr[-1])}
    w=np.zeros(2)
    for k in range(7):w-=m.ETA*m.H*(w-m.WML)
    err=float(np.max(abs(w-m.early_point(7))));assert err<1e-12
    assert not np.allclose(m.early_point(7),m.ridge_point(1/(7*m.ETA)))
    results['quadratic_iteration_max_error']=err
    x=np.array([.6,.8]);h=1e-6
    derivative=(m.radial(m.rotate(x,h),.7)-m.radial(m.rotate(x,-h),.7))/(2*h)
    err=float(abs(derivative-m.sensitivity(x,.7)));assert err<1e-8
    assert abs(m.sensitivity(x,0))<1e-14
    results['tangent_chain_rule_max_error']=err
    rows=[]
    for eps in [.6,.2,.04]:
        exact,approx=m.noise_example(eps)
        assert abs(exact-approx-.5*eps**4)<1e-14
        rows.append(dict(epsilon=eps,exact=exact,second_order=approx,error=exact-approx))
    results['noise_expansion']=rows
    original=m.convolution(m.IMAGE);shift=m.convolution(np.roll(m.IMAGE,1,axis=1))
    err=float(np.max(abs(original[:,:-1]-shift[:,1:])));assert err==0
    assert original[2,3]==np.sum(m.IMAGE[2:5,3:6]*m.KERNEL)
    pooled=m.subsample(m.sigmoid(original))
    assert abs(pooled[0,0]-m.sigmoid(m.sigmoid(original)[:2,:2].mean()))<1e-14
    results['cnn_shift_max_error']=err
    v=np.array([-.5,.0,.7])
    num=finite_gradient(lambda w:-np.log(m.mixture_components(w).sum(axis=1)).sum(),v)
    err=float(np.max(abs(num-m.soft_gradient(v))));assert err<1e-8
    assert np.allclose(m.responsibilities(v).sum(axis=1),1)
    results['soft_weight_gradient_max_error']=err
    rows=m.soft_history();target=rows[0][0]
    def loss(row):
        w,mu,s,p=row
        return 2*np.sum((w-target)**2)-np.log(m.mixture_components(w,mu,s,p).sum(1)).sum()
    assert loss(rows[-1])<loss(rows[0])
    results['joint_objective']=[float(loss(rows[0])),float(loss(rows[-1]))]
    results['checks_passed']=10
    Path('numerical_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
