"""PRML 5.4: linked curvature experiments, Manim Community Edition."""
import json
from pathlib import Path
import numpy as np
from manim import *
from scene_support import *
from hessian_model import *

BLUE=BLUE_CLASS; RED=RED_CLASS; GREEN=GREEN_CLASS; YELLOW=YELLOW_ACC; PURPLE=PURPLE_ACC

def path(ax,pts,color=BLUE,width=3):
    points=np.asarray(pts);o=ax.c2p(0,0)
    coords=o+points[:,0,None]*(ax.c2p(1,0)-o)+points[:,1,None]*(ax.c2p(0,1)-o)
    return VMobject().set_points_as_corners(coords).set_stroke(color,width)

def contours(ax,H,color=BLUE,levels=(.5,1.5,3)):
    return VGroup(*[path(ax,ellipse_points(H,level),color,2) for level in levels])

def matrix(fn,at=(3,0,0),color=BLUE):
    values=np.asarray(fn()); rows,cols=values.shape
    cells=VGroup()
    for i in range(rows):
        for j in range(cols):
            d=DecimalNumber(values[i,j],num_decimal_places=2,font_size=29,color=color)
            anchor=np.array(at)+np.array([(j-(cols-1)/2)*1.3,((rows-1)/2-i)*.7,0])
            d.move_to(anchor)
            d.add_updater(lambda m,i=i,j=j,a=anchor:m.set_value(fn()[i,j]).move_to(a))
            cells.add(d)
    brackets=VGroup(tex('[',70,color),tex(']',70,color))
    brackets[0].move_to(np.array(at)+LEFT*(cols*.65+.1));brackets[1].move_to(np.array(at)+RIGHT*(cols*.65+.1))
    return VGroup(cells,brackets)

def flow(edges,color,reverse=False):
    ordered=list(edges)[::-1] if reverse else list(edges)
    traces=[Line(e.get_end(),e.get_start(),color=color,stroke_width=5) if reverse else Line(e.get_start(),e.get_end(),color=color,stroke_width=5) for e in ordered]
    return LaggedStart(*[ShowPassingFlash(e,time_width=.55) for e in traces],lag_ratio=.5)

def vector(ax,p,color=YELLOW,origin=(0,0)):
    return Arrow(ax.c2p(*origin),ax.c2p(*(np.asarray(origin)+np.asarray(p))),buff=0,color=color,stroke_width=4,max_tip_length_to_length_ratio=.16)

class PRML54HessianMatrix(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.entries={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.question,self.directions,self.diagonal,self.outer,self.inverse,self.differences,self.exact,self.product,self.choose]):
            self.begin(i);method()
            assert self.bi==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        Path(config.media_dir,'prml54_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def body_clear(self):
        # Keep title and current captions; formulas are replaced explicitly.
        keep=[self.title,self.subtitle,self.formula]
        self.remove(*[m for m in self.mobjects if all(m is not k for k in keep)])

    def weight_axes(self):
        return self.plot_axes(x=(-2,2,1),y=(-2,2,1),width=3.8,height=3.8,center=(-2.6,-.1,0),labels=('w_1','w_2'))

    def question(self):
        k=ValueTracker(2);s=ValueTracker(0)
        ax=self.plot_axes(x=(-1.3,.6,.5),y=(0,3,1),width=8,height=3.4,labels=('s','E'))
        fn=lambda x:1+x+.5*k.get_value()*x*x
        baseline=curve(ax,lambda x:1+x+.5*x*x,color=MUTED)
        graph=always_redraw(lambda:curve(ax,fn,color=BLUE))
        dot=always_redraw(lambda:Dot(ax.c2p(s.get_value(),fn(s.get_value())),color=YELLOW))
        tangent=always_redraw(lambda:path(ax,[(x,fn(s.get_value())+(1+k.get_value()*s.get_value())*(x-s.get_value())) for x in [s.get_value()-.18,s.get_value()+.18]],YELLOW))
        control=knob('k=',k,1,4)
        self.add(baseline,graph,dot,tangent,control)
        self.equation(r'E(s)=1+s+\tfrac12ks^2',r'\quad E\prime(0)=1')
        self.beat(actions=[lambda:Create(baseline),lambda:s.animate.set_value(-1)])
        s.set_value(0)
        self.beat(actions=[lambda:k.animate.set_value(4),lambda:s.animate.set_value(-1)])
        self.equation(r'E\prime(s)=1+ks',r'\qquad E\prime\prime(s)=k')
        self.beat(actions=[lambda:s.animate.set_value(.35),lambda:Indicate(tangent,color=YELLOW)])
        self.body_clear()
        self.equation(r'w=(w_1,\ldots,w_W)^{\mathsf T}',r'\quad g=\nabla E')
        grid=Matrix([[r'\partial g_1/\partial w_1',r'\partial g_1/\partial w_2'],[r'\partial g_2/\partial w_1',r'\partial g_2/\partial w_2']],h_buff=3.8,v_buff=1.3,bracket_h_buff=.45).scale(.65).move_to([0,.2,0])
        grid.get_entries()[0].set_color(BLUE);grid.get_entries()[3].set_color(BLUE)
        grid.get_entries()[1].set_color(RED);grid.get_entries()[2].set_color(RED)
        self.beat(actions=[lambda:Write(grid),lambda:Indicate(grid,color=YELLOW)])
        self.equation(r'H_{ij}=\frac{\partial^2 E}{\partial w_i\partial w_j}',r'\quad (5.78)')
        self.beat(actions=[lambda:Indicate(grid.get_rows()[0],scale_factor=1.04),lambda:Indicate(VGroup(grid.get_entries()[1],grid.get_entries()[2]),color=RED)])
        uses=VGroup(*[jp(t,25,c) for t,c in [('学習の一歩',BLUE),('再学習',GREEN),('刈り込み',RED),('不確かさ',PURPLE)]]).arrange(RIGHT,buff=.7).move_to([0,-1.7,0])
        self.beat(actions=[lambda:LaggedStart(*[FadeIn(x,shift=UP*.2) for x in uses[:3]],lag_ratio=.25),lambda:FadeIn(uses[3],shift=UP*.2)])

    def directions(self):
        ax=self.weight_axes();H=np.array([[3.,2.],[2.,3.]])
        theta=ValueTracker(-PI/4)
        direction=lambda:np.array([np.cos(theta.get_value()),np.sin(theta.get_value())])
        rings=contours(ax,H);arrow=always_redraw(lambda:vector(ax,1.4*direction()))
        curvature=number(r'u^{\mathsf T}Hu=',lambda:direction()@H@direction(),(3,.6,0),YELLOW)
        self.add(rings,arrow,curvature,note('同じ距離でも、方向で増え方が変わる'))
        self.equation(r'E(w)=\tfrac12w^{\mathsf T}Hw',r'\quad H=\begin{bmatrix}3&2\\2&3\end{bmatrix}')
        self.beat(actions=[lambda:Create(rings),lambda:theta.animate.set_value(PI/4)])
        self.beat(actions=[lambda:theta.animate.set_value(3*PI/4),lambda:theta.animate.set_value(PI/4)])
        self.equation(r'Hu_i=\lambda_i u_i',r'\qquad \lambda_1=1,\ \lambda_2=5')
        eigen=VGroup(vector(ax,[1.2,-1.2],GREEN),vector(ax,[.6,.6],RED))
        self.beat(actions=[lambda:Create(eigen),lambda:theta.animate.set_value(-PI/4)])
        self.equation(r'E(w+d)\simeq E(w)+g^{\mathsf T}d+\tfrac12d^{\mathsf T}Hd',size=30)
        self.beat(actions=[lambda:Indicate(self.formula),lambda:Indicate(Dot(ax.c2p(0,0),color=GREEN))])
        self.body_clear();lam=ValueTracker(1)
        ax=self.plot_axes(x=(-1.5,1.5,.5),y=(-1.5,2,1),width=7,height=3.2,labels=('s',r'E-E_0'))
        cross=always_redraw(lambda:curve(ax,lambda x:.5*lam.get_value()*x*x,color=RED))
        other=curve(ax,lambda x:.5*x*x,color=BLUE)
        self.add(other,cross,number(r'\lambda=',lam.get_value,(4,.2,0),RED))
        self.equation(r'g=0:\quad E-E_0=\tfrac12\lambda s^2')
        self.beat(actions=[lambda:lam.animate.set_value(-1),lambda:Indicate(other,color=BLUE)])
        self.equation(r'g=0,\ \lambda_i>0\ \forall i\Rightarrow\text{local minimum}',size=30)
        self.beat(actions=[lambda:lam.animate.set_value(1),lambda:Transform(self.formula,tex(r'H\in\mathbb{R}^{W\times W}:\quad W^2',36).move_to([0,2.48,0]))])

    def diagonal(self):
        ax=self.weight_axes();c=ValueTracker(2)
        H=lambda:np.array([[3,c.get_value()],[c.get_value(),3]])
        rings=always_redraw(lambda:contours(ax,H()))
        mat=matrix(H);self.add(rings,mat,knob('c=',c,0,2))
        self.equation(r'H=\begin{bmatrix}3&c\\c&3\end{bmatrix}')
        self.beat(actions=[lambda:c.animate.set_value(0),lambda:Indicate(mat,color=RED)])
        self.beat(actions=[lambda:c.animate.set_value(2),lambda:Indicate(rings,color=YELLOW)])
        self.equation(r'D^{-1}=\mathrm{diag}(1/D_{11},\ldots,1/D_{WW})')
        self.beat(actions=[lambda:c.animate.set_value(0),lambda:Indicate(self.formula,color=RED)])
        self.equation(r'\frac{\partial^2 E_n}{\partial w_{ji}^2}=z_i^2\frac{\partial^2 E_n}{\partial a_j^2}\quad(5.79)')
        self.beat(actions=[lambda:Indicate(self.formula,color=BLUE),lambda:c.animate.set_value(1)])
        self.equation(r'q_j\simeq[h\prime(a_j)]^2\sum_k w_{kj}^2q_k+h\prime\prime(a_j)\sum_k w_{kj}\delta_k',size=28)
        labels=VGroup(tex(r'q_j=\partial^2E_n/\partial a_j^2',25),tex(r'\delta_k=\partial E_n/\partial a_k',25)).arrange(DOWN,buff=.25).move_to([3,-1.4,0])
        self.add(labels)
        self.beat(actions=[lambda:c.animate.set_value(0),lambda:Indicate(labels,color=YELLOW)])
        self.equation(r'\text{recursive diagonal approximation}: O(W)\qquad (5.81)',size=28)
        self.beat(actions=[lambda:c.animate.set_value(2),lambda:c.animate.set_value(0)])

    def outer(self):
        r=ValueTracker(.7);w=WEIGHTS.copy();y=net(w)['y']
        ax=self.plot_axes(x=(-1.5,1.5,.5),y=(-1.2,1.2,.5),width=7.3,height=3.2,labels=('x','y,t'))
        graph=curve(ax,lambda x:w[1]*np.tanh(w[0]*x),color=BLUE)
        prediction=Dot(ax.c2p(X,y),color=BLUE)
        target=always_redraw(lambda:Dot(ax.c2p(X,y-r.get_value()),color=RED))
        residual=always_redraw(lambda:Line(ax.c2p(X,y),ax.c2p(X,y-r.get_value()),color=RED,stroke_width=5))
        self.add(graph,prediction,target,residual,knob('r=y-t=',r,0,.7))
        self.equation(r'z=\tanh(ux),\ y=vz,\quad E=\tfrac12(y-t)^2\quad(5.82)',size=30)
        self.beat(actions=[lambda:Create(graph),lambda:Indicate(residual,color=RED)])
        self.beat(actions=[lambda:r.animate.set_value(.03),lambda:Indicate(target,color=RED)])
        self.body_clear();r.set_value(.7)
        outer=matrix(lambda:net(w)['outer'],(-2.8,.2,0),BLUE)
        residual_matrix=matrix(lambda:net(w,t=y-r.get_value())['residual'],(2.8,.2,0),RED)
        self.add(outer,residual_matrix,tex('+',48).move_to([0,.2,0]),knob('r=',r,0,.7))
        self.add(jp('勾配の外積',25,BLUE).move_to([-2.8,1.35,0]),jp('残差 × 出力の二階微分',23,RED).move_to([2.8,1.35,0]))
        eq=self.equation(r'H=',r'\sum_n b_nb_n^{\mathsf T}',r'+',r'\sum_n r_n\nabla^2y_n',r'\quad b_n=\nabla y_n',size=30)
        eq[1].set_color(BLUE);eq[3].set_color(RED)
        self.beat(actions=[lambda:Indicate(outer,color=BLUE),lambda:r.animate.set_value(0)])
        self.beat(actions=[lambda:Indicate(eq[1],color=BLUE),lambda:r.animate.set_value(.7)])
        self.beat(actions=[lambda:r.animate.set_value(0),lambda:r.animate.set_value(.18)])
        self.body_clear();p=ValueTracker(.05)
        self.equation(r'H\simeq\sum_n y_n(1-y_n)b_nb_n^{\mathsf T}',r'\quad b_n=\nabla a_n\quad(5.85)',size=29)
        ax=self.plot_axes(x=(0,1,.25),y=(0,.3,.1),height=3,width=7,labels=('y','y(1-y)'))
        graph=curve(ax,lambda x:x*(1-x),color=BLUE)
        dot=always_redraw(lambda:Dot(ax.c2p(p.get_value(),p.get_value()*(1-p.get_value())),color=YELLOW))
        self.add(graph,dot,note('sigmoid ＋ 二値交差エントロピー'))
        self.beat(actions=[lambda:p.animate.set_value(.5),lambda:p.animate.set_value(.95)])

    def inverse(self):
        n=ValueTracker(0);ax=self.weight_axes()
        rings=always_redraw(lambda:contours(ax,precision(n.get_value()),levels=(.3,.6,1)))
        mat=matrix(lambda:inverse_update(n.get_value()),(3,.1,0),PURPLE)
        arrow=always_redraw(lambda:vector(ax,B[min(int(n.get_value()),2)],YELLOW))
        self.add(rings,mat,arrow,knob('L=',n,0,3),jp('逆行列',25,PURPLE).move_to([3,1.35,0]))
        self.equation(r'H_0=\alpha I,\quad H_0^{-1}=\alpha^{-1}I,\quad\alpha=0.3')
        self.beat(actions=[lambda:Create(rings),lambda:Indicate(mat,color=PURPLE)])
        self.equation(r'H_{L+1}=H_L+b_{L+1}b_{L+1}^{\mathsf T}\quad(5.87)')
        self.beat(actions=[lambda:n.animate.set_value(1),lambda:Indicate(rings,color=PURPLE)])
        self.beat(actions=[lambda:n.animate.set_value(2),lambda:n.animate.set_value(2.3)])
        self.equation(r'C_{L+1}=C_L-\frac{C_Lbb^{\mathsf T}C_L}{1+b^{\mathsf T}C_Lb},\quad C_L=H_L^{-1}\quad(5.89)',size=29)
        self.beat(actions=[lambda:n.animate.set_value(3),lambda:Indicate(mat,color=PURPLE)])
        self.equation(r'C_N=\left(\alpha I+\sum_{n=1}^N b_nb_n^{\mathsf T}\right)^{-1}')
        self.beat(actions=[lambda:Indicate(self.formula,color=YELLOW),lambda:Indicate(mat,color=PURPLE)])
        scan=ValueTracker(0)
        point=always_redraw(lambda:Dot(ax.c2p(*ellipse_points(precision(3),1,181)[int(scan.get_value())%181]),color=YELLOW))
        self.add(point)
        self.beat(actions=[lambda:scan.animate.set_value(90),lambda:scan.animate.set_value(180)])

    def differences(self):
        eps=ValueTracker(.35);ax=self.plot_axes(x=(-.5,.5,.25),y=(-.5,.5,.25),width=4,height=3.3,center=(-2.5,-.15,0),labels=(r'\Delta u',r'\Delta v'))
        dots=always_redraw(lambda:VGroup(*[VGroup(Dot(ax.c2p(s*eps.get_value(),t*eps.get_value()),color=BLUE if s*t>0 else RED),tex('+' if s*t>0 else '-',23).move_to(ax.c2p(s*eps.get_value(),t*eps.get_value())+UP*.22)) for s,t in [(1,1),(1,-1),(-1,1),(-1,-1)]]))
        self.add(dots,number(r'\widehat H_{uv}=',lambda:finite_hessian(WEIGHTS,eps.get_value())[0,1],(3,.4,0),YELLOW,places=5),tex(r'H_{uv}='+f"{net()['H'][0,1]:.5f}",28,BLUE).move_to([3,-.5,0]))
        self.equation(r'H_{uv}\simeq\frac{E_{++}-E_{+-}-E_{-+}+E_{--}}{4\epsilon^2}\quad(5.90)',size=30)
        self.beat(actions=[lambda:Create(dots),lambda:Indicate(dots,color=YELLOW)])
        self.beat(actions=[lambda:eps.animate.set_value(.04),lambda:eps.animate.set_value(.015)])
        self.body_clear()
        ax=self.plot_axes(x=(-8,-1,1),y=(-10,0,2),width=8.2,height=3.1,labels=(r'\log_{10}\epsilon',r'\log_{10}|\mathrm{error}|'))
        powers=np.linspace(-8,-1,100)
        errors=np.array([max(abs(finite_hessian(WEIGHTS,10**p)[0,1]-net()['H'][0,1]),1e-11) for p in powers])
        points=np.column_stack([powers,np.log10(errors)])
        line=path(ax,points,RED);self.add(line,note('実際の浮動小数点計算：小さすぎる幅で誤差が増える'))
        scan=ValueTracker(99)
        dot=always_redraw(lambda:Dot(ax.c2p(*points[int(scan.get_value())]),color=YELLOW));self.add(dot)
        self.beat(actions=[lambda:scan.animate.set_value(0),lambda:scan.animate.set_value(58)])
        self.equation(r'H_{:j}\simeq\frac{g(w+\epsilon e_j)-g(w-\epsilon e_j)}{2\epsilon}\quad(5.91)',size=30)
        self.beat(actions=[lambda:Indicate(self.formula,color=BLUE),lambda:scan.animate.set_value(70)])
        self.body_clear();self.cost_plot(cubic=True)
        self.beat(actions=[lambda:self.cost_tracker.animate.set_value(9),lambda:Indicate(self.cost_lines[1],color=BLUE)])
        self.equation(r'\epsilon=10^{-4}:\quad H_{uv}\approx '+f"{finite_hessian(WEIGHTS)[0,1]:.6f}"+r',\quad H_{uv}='+f"{net()['H'][0,1]:.6f}",size=30)
        self.beat(actions=[lambda:Indicate(self.formula,color=GREEN),lambda:self.cost_tracker.animate.set_value(3)])

    def network(self):
        nodes=VGroup(*[Circle(.31,color=c,fill_color=c,fill_opacity=.12).move_to([x,.75,0]) for x,c in [(-4,BLUE),(-1.3,GREEN),(1.5,PURPLE),(4,RED)]])
        labels=VGroup(*[tex(t,26,c).move_to(n) for n,t,c in zip(nodes,['x','a','z','y'],[BLUE,GREEN,PURPLE,RED])])
        edges=VGroup(*[Arrow(a.get_right(),b.get_left(),buff=.06,color=MUTED) for a,b in zip(nodes,nodes[1:])])
        edge_labels=VGroup(*[tex(t,25,c).next_to(e,UP,buff=.12) for e,t,c in zip(edges,['u',r'h=\tanh','v'],[BLUE,GREEN,RED])])
        return VGroup(edges,nodes,labels,edge_labels)

    def exact(self):
        u=ValueTracker(.7);w=lambda:np.array([u.get_value(),1.2])
        network=self.network();self.add(network)
        self.equation(r'a=ux,\quad z=h(a),\quad y=vz,\quad E=\tfrac12(y-t)^2',size=30)
        mat=matrix(lambda:net(w())['H'],(0,-1.05,0),BLUE)
        self.add(mat,jp('行・列の順序：u, v',21,MUTED).move_to([3.7,-1.1,0]),knob('u=',u,.2,1.2))
        self.beat(actions=[lambda:flow(network[0],YELLOW),lambda:Indicate(network[3][1],color=GREEN)])
        self.equation(r'\delta=y-t,\ M=1:\quad H_{vv}=z^2 M=z^2\quad(5.92),(5.93)',size=29)
        self.beat(actions=[lambda:Indicate(network[3][2],color=RED),lambda:Indicate(mat[0][3],color=RED)])
        self.equation(r'H_{uu}=x^2\left[(vh\prime)^2+\delta v h\prime\prime\right]\quad(5.94)',size=30)
        self.beat(actions=[lambda:Indicate(network[3][0],color=BLUE),lambda:Indicate(mat[0][0],color=BLUE)])
        self.equation(r'H_{uv}=H_{vu}=xh\prime(vz+\delta)\quad(5.95)',size=32)
        self.beat(actions=[lambda:Indicate(network[0],color=YELLOW,scale_factor=1),lambda:Indicate(VGroup(mat[0][1],mat[0][2]),color=YELLOW)])
        self.equation(r'H=bb^{\mathsf T}+r\nabla^2 y,\quad b=\nabla y')
        self.beat(actions=[lambda:u.animate.set_value(.2),lambda:u.animate.set_value(1.1)])
        self.equation(r'\text{bias}: x_0=z_0=1,\qquad H\ \text{full}:O(W^2)',size=30)
        self.beat(actions=[lambda:u.animate.set_value(.7),lambda:Indicate(mat,color=GREEN)])

    def product(self):
        angle=ValueTracker(.7);direction=lambda:np.array([np.cos(angle.get_value()),np.sin(angle.get_value())])
        step=ValueTracker(0);ax=self.plot_axes(x=(0,.6,.2),y=(0,.6,.2),width=3.6,height=3.6,center=(-2.1,-.1,0),labels=(r'g_u',r'g_v'))
        old=net()['g'];arrow=always_redraw(lambda:vector(ax,net(WEIGHTS+step.get_value()*direction())['g'],BLUE))
        fixed=vector(ax,old,MUTED);delta=always_redraw(lambda:vector(ax,net(WEIGHTS+step.get_value()*direction())['g']-old,YELLOW,origin=old))
        self.add(fixed,arrow,delta,note('青：移動後の勾配　黄：勾配の変化'),number(r'\epsilon=',step.get_value,(3,.5,0),YELLOW),number(r'\|\Delta g\|=',lambda:np.linalg.norm(net(WEIGHTS+step.get_value()*direction())['g']-old),(3,-.4,0),YELLOW,places=3))
        self.equation(r'g(w+\epsilon v)-g(w)\simeq\epsilon Hv\quad(5.96)')
        self.beat(actions=[lambda:step.animate.set_value(.3),lambda:step.animate.set_value(.05)])
        self.equation(r'R\{f\}=\left.\frac{d f(w+\epsilon v)}{d\epsilon}\right|_{\epsilon=0},\quad R\{w\}=v\quad(5.97)',size=29)
        self.beat(actions=[lambda:step.animate.set_value(.2),lambda:step.animate.set_value(.01)])
        self.body_clear();network=self.network();self.add(network)
        d=direction();result,vals=hvp(WEIGHTS,d)
        forward=VGroup(*[VGroup(tex(k,27,GREEN),DecimalNumber(vals[k],num_decimal_places=3,font_size=27,color=GREEN)).arrange(RIGHT,buff=.15) for k in ['Ra','Rz','Ry']]).arrange(RIGHT,buff=.65).move_to([0,-.35,0])
        self.equation(r'Ra=v_u x,\quad Rz=h\prime Ra,\quad Ry=v_vz+vRz\quad(5.101)\text{–}(5.103)',size=28)
        self.add(VGroup(jp('方向の成分',22),tex(r'v_u,v_v',27),jp('出力重み',22),tex('v',27)).arrange(RIGHT,buff=.25).move_to([0,-2.45,0]))
        self.beat(actions=[lambda:AnimationGroup(Write(forward),flow(network[0],GREEN)),lambda:Indicate(forward,color=GREEN)])
        self.remove(forward)
        delta_definition=tex(r'\delta=y-t,\quad\delta_h=h\prime v\delta',27,RED).move_to([0,-1.5,0])
        self.add(delta_definition)
        backward=VGroup(tex(r'R\delta=Ry='+f"{vals['Rdelta']:.3f}",28,RED),tex(r'R\delta_h='+f"{vals['Rhidden']:.3f}",28,RED)).arrange(RIGHT,buff=1).move_to([0,-.35,0])
        self.equation(r'R\delta_h=h\prime\prime Ra\,v\delta+h\prime v_v\delta+h\prime vR\delta',size=29)
        self.beat(actions=[lambda:AnimationGroup(Write(backward),flow(network[0],RED,reverse=True)),lambda:Indicate(self.formula,color=RED)])
        self.remove(backward,delta_definition)
        self.equation(r'(Hv)_u=xR\delta_h,\quad(Hv)_v=R\delta\,z+\delta Rz\quad(5.110),(5.111)',size=29)
        output=matrix(lambda:hvp(WEIGHTS,direction())[0][:,None],(-2,-.5,0),YELLOW)
        direct=matrix(lambda:(net()['H']@direction())[:,None],(2,-.5,0),BLUE)
        self.add(output,direct,tex('=',32).move_to([0,-.5,0]),jp('R の再帰',22,YELLOW).move_to([-2,-1.65,0]),jp('行列を作った積',22,BLUE).move_to([2,-1.65,0]))
        self.beat(actions=[lambda:Indicate(output,color=YELLOW),lambda:Indicate(self.formula,color=YELLOW)])
        self.body_clear();ax=self.plot_axes(x=(-1.2,1.2,.5),y=(-1.2,1.2,.5),width=3.6,height=3.6,center=(0,-.1,0),labels=(r'v_u',r'v_v'))
        v=always_redraw(lambda:vector(ax,direction(),BLUE));hv=always_redraw(lambda:vector(ax,hvp(WEIGHTS,direction())[0],YELLOW))
        self.add(v,hv,note('青：方向 v　黄：勾配の変化率 Hv'))
        self.equation(r'R\{\nabla E\}=Hv,\quad (v^{\mathsf T}H)^{\mathsf T}=Hv')
        self.beat(actions=[lambda:angle.animate.set_value(PI),lambda:angle.animate.set_value(2*PI+.7)])

    def cost_plot(self,cubic=False):
        ax=self.plot_axes(x=(1,10,3),y=(0,3 if cubic else 2,1),width=7.6,height=3,labels=('W',r'\log_{10}(\mathrm{cost})'))
        powers=[3,2] if cubic else [2,1]
        lines=VGroup(*[curve(ax,lambda w,p=p:p*np.log10(w),color=c) for p,c in zip(powers,[RED,BLUE])])
        t=ValueTracker(2);dots=always_redraw(lambda:VGroup(*[Dot(ax.c2p(t.get_value(),p*np.log10(t.get_value())),color=c) for p,c in zip(powers,[RED,BLUE])]))
        names=VGroup(*[tex(f'O(W^{p})' if p>1 else 'O(W)',26,c) for p,c in zip(powers,[RED,BLUE])]).arrange(DOWN,buff=.4).move_to([5,.2,0])
        self.add(lines,dots,names,note('一例あたりの計算量：実測時間ではない'))
        self.cost_tracker=t;self.cost_lines=lines

    def choose(self):
        count=ValueTracker(2)
        def cells():
            n=int(round(count.get_value()));size=2.8/n
            full=VGroup(*[Square(size*.85,color=BLUE,fill_opacity=.3).move_to([-2.5+(j-(n-1)/2)*size,.2+(i-(n-1)/2)*size,0]) for i in range(n) for j in range(n)])
            vec=VGroup(*[Square(size*.85,color=YELLOW,fill_opacity=.4).move_to([3,.2+(i-(n-1)/2)*size,0]) for i in range(n)])
            return VGroup(full,vec)
        grid=always_redraw(cells);self.add(grid,number('W=',count.get_value,(0,-2.2,0),places=0))
        self.equation(r'H:W^2\ \text{entries}',r'\qquad Hv:W\ \text{entries}')
        self.beat(actions=[lambda:count.animate.set_value(8),lambda:Indicate(grid,color=YELLOW)])
        self.body_clear();self.cost_plot()
        self.equation(r'H:O(W^2),\qquad Hv:O(W)')
        self.beat(actions=[lambda:self.cost_tracker.animate.set_value(9),lambda:Indicate(self.cost_lines[0],color=RED)])
        self.body_clear();H=net()['H'];mat=matrix(lambda:H,(2.4,-.1,0));self.add(mat)
        self.equation(r'He_j=H_{:j},\qquad j=1,\ldots,W')
        v1=tex(r'e_1=\begin{bmatrix}1\\0\end{bmatrix}',36,YELLOW).move_to([-2,-.1,0]);self.add(v1)
        self.beat(actions=[lambda:Indicate(VGroup(mat[0][0],mat[0][2]),color=YELLOW),lambda:AnimationGroup(Transform(v1,tex(r'e_2=\begin{bmatrix}0\\1\end{bmatrix}',36,YELLOW).move_to(v1)),Indicate(VGroup(mat[0][1],mat[0][3]),color=YELLOW))])
        self.equation(r'\mathrm{diag}(H),\quad \sum_n b_nb_n^{\mathsf T},\quad R\{\nabla E\}',size=35)
        self.beat(actions=[lambda:Indicate(VGroup(mat[0][1],mat[0][2]),color=RED),lambda:Indicate(mat,color=GREEN)])
        self.body_clear();k=ValueTracker(1)
        ax=self.plot_axes(x=(-1.3,.6,.5),y=(0,3,1),width=7.5,height=3.2,labels=('s','E'))
        graph=always_redraw(lambda:curve(ax,lambda x:1+x+.5*k.get_value()*x*x,color=BLUE))
        self.add(graph,knob('k=',k,1,4))
        self.equation(r'E\prime(0)=1,\qquad E\prime\prime(0)=k')
        self.beat(actions=[lambda:k.animate.set_value(4),lambda:k.animate.set_value(1)])
        self.equation(r'\text{gradient}\quad\longrightarrow\quad\text{curvature}\quad\longrightarrow\quad Hv',size=34)
        self.beat(actions=[lambda:k.animate.set_value(4),lambda:k.animate.set_value(2)])
