"""Marching squares for probability isolines, with bilinear saddle disambiguation."""
import numpy as np

def contours(x,y,z,level):
    # Vertices clockwise: bottom left, bottom right, top right, top left.
    corners=np.stack([z[:-1,:-1],z[:-1,1:],z[1:,1:],z[1:,:-1]],axis=-1)
    above=corners>=level
    crossing=above != np.roll(above,-1,axis=-1)
    rows,cols=np.where(crossing.any(axis=-1))
    segments=[]
    for r,c in zip(rows,cols):
        verts=np.array([[x[c],y[r]],[x[c+1],y[r]],[x[c+1],y[r+1]],[x[c],y[r+1]]])
        val=corners[r,c];edges=np.flatnonzero(crossing[r,c]);pts={}
        for e in edges:
            n=(e+1)%4;f=(level-val[e])/(val[n]-val[e]);pts[e]=verts[e]+f*(verts[n]-verts[e])
        if len(edges)==2:segments.append(np.array([pts[e] for e in edges]))
        elif len(edges)==4:
            # Asymptotic decider for saddle cells, deterministic at equality.
            q=(val[0]-level)*(val[2]-level)-(val[1]-level)*(val[3]-level)
            pairs=[(0,1),(2,3)] if q>=0 else [(0,3),(1,2)]
            segments.extend(np.array([pts[a],pts[b]]) for a,b in pairs)
    return segments
