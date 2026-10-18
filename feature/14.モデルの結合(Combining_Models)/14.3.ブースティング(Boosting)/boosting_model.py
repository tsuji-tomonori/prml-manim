"""A reproducible ten-point, three-round AdaBoost illustration for PRML 14.3."""
import numpy as np
X=np.arange(10,dtype=float)+.5
T=np.array([-1,-1,1,1,-1,-1,1,1,1,1],dtype=int)
QUERY_INDEX=4

def stump(k,left):
    return np.where(np.arange(len(X))<k,left,-left).astype(int)

def rounds(count=3):
    w=np.ones(len(X),dtype=float)/len(X)
    states=[]
    score=np.zeros(len(X))
    for m in range(count):
        candidates=[]
        for k in range(len(X)+1):
            for left in (-1,1):
                y=stump(k,left)
                epsilon=float(w[y!=T].sum()/w.sum())
                candidates.append((round(epsilon,12),k,left,y))
        epsilon,k,left,y=min(candidates,key=lambda c:c[:3])
        if not 0<epsilon<.5:
            raise ValueError('This pedagogical example needs a weak classifier with 0 < epsilon < 0.5')
        alpha=float(np.log((1-epsilon)/epsilon))
        before=w.copy()
        unnormalized=w*np.exp(alpha*(y!=T))
        w=unnormalized/unnormalized.sum()
        score=score+alpha*y
        states.append(dict(m=m+1,k=k,left=left,epsilon=epsilon,alpha=alpha,
                           prediction=y.tolist(),weight_before=before.tolist(),
                           weight_unnormalized=unnormalized.tolist(),weight_after=w.tolist(),
                           score=score.tolist(),committee=np.sign(score).astype(int).tolist()))
    return states

if __name__=='__main__':
    for s in rounds():
        print(s['m'],s['k'],s['left'],s['epsilon'],s['alpha'],s['committee'])
