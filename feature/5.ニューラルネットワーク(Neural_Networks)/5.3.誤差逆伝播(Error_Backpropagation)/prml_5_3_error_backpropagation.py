"""PRML 5.3: a moving numerical experiment, in Manim Community Edition."""
import json
from pathlib import Path
import numpy as np
from manim import *
from narration_content import SCENES
from make_voicevox_narration import MANIFEST
from scene_support import *
from backprop_model import *

ROOT=Path(__file__).parent
C=[BLUE_CLASS,GREEN_CLASS,RED_CLASS]

def pulse(paths,color=RED_CLASS):
    return LaggedStart(*[ShowPassingFlash(p.copy().set_color(color).set_stroke(width=6),time_width=.5) for p in paths],lag_ratio=.16)

def tangent(ax,x,y,slope,color=RED_CLASS,half=.35):
    return Line(ax.c2p(x-half,y-half*slope),ax.c2p(x+half,y+half*slope),color=color,stroke_width=4)

def network(getter,scale=1,center=(0,0,0),deltas=False):
    """Same two-layer graph for forward activations and backward sensitivities."""
    nodes=VGroup(); labels=VGroup(); values=VGroup();layers=[]
    for l,key in enumerate(['x','z','y']):
        layer=VGroup()
        for j in range(2):
            pos=np.array([-4+4*l,.85-1.7*j,0])*scale+np.array(center)
            circle=Circle(radius=.39*scale,color=C[l],fill_color=BG,fill_opacity=1).move_to(pos)
            circle.add_updater(lambda m,k=key,i=j:m.set_fill(C[['x','z','y'].index(k)],opacity=.08+.25*min(abs(getter()[k][i+(k=='x')]),1)))
            symbol=('x',r'\delta' if deltas else 'z',r'\delta' if deltas else 'y')[l]
            labels.add(tex(fr'{symbol}_{{{j+1}}}',26*scale,C[l]).next_to(circle,UP,buff=.12))
            gkey=('x','d1','d2')[l] if deltas else key
            val=DecimalNumber(getter()[gkey][j+(gkey=='x')],num_decimal_places=2,font_size=23*scale,color=WHITE).move_to(pos)
            val.add_updater(lambda m,k=gkey,i=j,p=pos:m.set_value(getter()[k][i+(k=='x')]).move_to(p))
            layer.add(circle);values.add(val)
        layers.append(layer);nodes.add(*layer)
    edges=[]
    for l,weights in enumerate([W1[:,1:],W2[:,1:]]):
        group=VGroup()
        for j in range(2):
            for i in range(2):
                a,b=layers[l][i].get_center(),layers[l+1][j].get_center()
                group.add(Line(a,b,buff=.4*scale,color=BLUE_CLASS if weights[j,i]>0 else RED_CLASS,stroke_width=1+2*abs(weights[j,i])))
        edges.append(group)
    group=VGroup(*edges,nodes,labels,values)
    return group,edges,layers

class PRML53ErrorBackpropagation(NarratedScene):
    def construct(self):
        self.camera.background_color=BG
        self.entries={e['id']:e for e in json.loads(MANIFEST.read_text())['scenes']}
        self.timeline=[]
        for i,method in enumerate([self.question,self.forward_pass,self.chain,self.output_delta,self.branch,self.activation,self.dataset,self.efficiency,self.input_jacobian]):
            self.begin(i);method()
            assert self.bi==len(self.story['beats'])
            self.timeline[-1]['end']=float(self.time)
        (ROOT/'media/prml53_timeline.json').write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2)+'\n')

    def question(self):
        w=ValueTracker(-.6)
        ax=self.plot_axes(x=(-1,1.7,.5),y=(0,1.25,.5),width=8,height=3.4,center=(-.7,-.1,0),labels=('w','E_n'))
        graph=curve(ax,lambda u:scalar(u)['E'],color=RED_CLASS)
        dot=always_redraw(lambda:Dot(ax.c2p(w.get_value(),scalar(w.get_value())['E']),color=YELLOW_ACC))
        read=number('y=',lambda:scalar(w.get_value())['y'],(4.7,.8,0),BLUE_CLASS)
        target=tex('t=0.70',28,GREEN_CLASS).move_to([4.7,0,0])
        k=knob('w=',w,-1,1.7,at=(-1,-2.5,0))
        self.equation(r'y=1.2\tanh(0.8w+0.1)-0.1,\quad E_n=\tfrac12(y-0.7)^2',size=29)
        self.add(graph,dot,read,target,k)
        self.beat(actions=[lambda:Indicate(target),lambda:Indicate(k[0])])
        self.beat(w.animate.set_value(1.4))
        self.beat(w.animate.set_value(-.25))
        tan=always_redraw(lambda:tangent(ax,w.get_value(),scalar(w.get_value())['E'],scalar(w.get_value())['g']))
        self.add(tan)
        self.beat(w.animate.set_value(.6))
        self.equation(r'\nabla E\quad\longrightarrow\quad\Delta\mathbf{w}')
        self.add(note('勾配を計算 → 重みを更新',at=(3.8,-.9,0)))
        self.beat(actions=[lambda:Circumscribe(tan),lambda:w.animate.set_value(.8)])

    def forward_pass(self):
        u=ValueTracker(.6); get=lambda:forward(np.array([u.get_value(),-.4]))
        net,edges,layers=network(get)
        self.add(net,note('自作例：辺の青は正、赤は負。バイアスの辺は省略。'))
        self.equation(r'\mathbf{x}=(0.6,-0.4)^T')
        self.beat(actions=[lambda:pulse(edges[0],BLUE_CLASS),lambda:Circumscribe(layers[1])])
        self.equation(r'a_j=\sum_i w_{ji}z_i',size=36)
        a=VGroup(*[number(fr'a_{j+1}=',lambda j=j:get()['a'][j],(-.1,1.7-.5*j,0),YELLOW_ACC,size=23) for j in range(2)])
        # Move the sums away from the hidden nodes.
        a.move_to([-2.2,1.55,0]); self.add(a)
        self.beat(actions=[lambda:pulse(edges[0],BLUE_CLASS),lambda:Circumscribe(a)])
        self.remove(a);self.equation(r'z_j=h(a_j)=\tanh(a_j)')
        self.beat(actions=[lambda:Circumscribe(layers[1]),lambda:u.animate.set_value(.9)])
        self.equation(r'y_k=\sum_{j=0}^{M}w^{(2)}_{kj}z_j')
        self.beat(actions=[lambda:pulse(edges[1],BLUE_CLASS),lambda:Circumscribe(layers[2])])
        self.beat(actions=[lambda:u.animate.set_value(-.7),lambda:u.animate.set_value(.6)])
        self.equation(r'x_0=z_0=1,\qquad a_j,\ z_j\ \longrightarrow\ \mathrm{save}')
        self.beat(actions=[lambda:Indicate(self.formula),lambda:LaggedStart(*[Circumscribe(l) for l in layers],lag_ratio=.3)])

    def chain(self):
        w=ValueTracker(.5);x=ValueTracker(.8); get=lambda:scalar(w.get_value(),x.get_value())
        coords=[-5,-2.5,0,2.5,5];names=['w','a','z','y','E_n'];colors=[YELLOW_ACC,BLUE_CLASS,GREEN_CLASS,BLUE_CLASS,RED_CLASS]
        groups=VGroup(); paths=VGroup()
        for j,(p,n,c) in enumerate(zip(coords,names,colors)):
            box=RoundedRectangle(width=1.6,height=1.3,corner_radius=.15,color=c).move_to([p,.35,0])
            label=tex(n,30,c).move_to([p,.65,0]);fn=w.get_value if j==0 else lambda n=n:get()['E' if n=='E_n' else n]
            value=DecimalNumber(fn(),num_decimal_places=3,font_size=27).move_to([p,.08,0])
            value.add_updater(lambda m,fn=fn,p=p:m.set_value(fn()).move_to([p,.08,0]))
            groups.add(VGroup(box,label,value))
            if j:paths.add(Arrow([coords[j-1]+.85,.35,0],[p-.85,.35,0],buff=0,color=MUTED))
        slider=knob('w=',w,.1,1.2,at=(-1,-2.5,0))
        self.add(groups,paths,slider)
        self.equation(r'a=wx+0.1,\quad z=\tanh(a),\quad y=1.2z-0.1')
        self.beat(actions=[lambda:w.animate.set_value(.65),lambda:pulse(paths,BLUE_CLASS)])
        local=VGroup(tex('x',28,BLUE_CLASS),tex('1-z^2',28,GREEN_CLASS),tex('1.2',28,BLUE_CLASS),tex('y-t',28,RED_CLASS))
        for m,p in zip(local,[-3.75,-1.25,1.25,3.75]):m.move_to([p,-.7,0])
        self.add(local)
        self.equation(r'\frac{\partial E_n}{\partial w}=\frac{\partial E_n}{\partial y}\frac{\partial y}{\partial z}\frac{\partial z}{\partial a}\frac{\partial a}{\partial w}',size=35)
        self.beat(actions=[lambda:Indicate(local[0]),lambda:LaggedStart(*[Indicate(m) for m in local],lag_ratio=.3)])
        self.equation(r'\delta_j\equiv\frac{\partial E_n}{\partial a_j}',size=39)
        brace=Brace(VGroup(*groups[1:]),DOWN,buff=.65,color=RED_CLASS)
        self.beat(actions=[lambda:Create(brace),lambda:Indicate(self.formula)])
        self.remove(brace);self.equation(r'\frac{\partial a_j}{\partial w_{ji}}=z_i',size=39)
        self.beat(actions=[lambda:w.animate.set_value(.8),lambda:x.animate.set_value(0)])
        self.equation(r'\frac{\partial E_n}{\partial w_{ji}}=',r'\delta_j',r'z_i',size=40)
        self.formula[1].set_color(RED_CLASS);self.formula[2].set_color(BLUE_CLASS)
        self.beat(actions=[lambda:x.animate.set_value(.8),lambda:Indicate(local[0])])
        self.equation(r'\delta_j\ \mathrm{fixed}:\quad z_i\mapsto-z_i\Rightarrow\delta_jz_i\mapsto-\delta_jz_i')
        # Hold downstream sensitivity fixed to isolate the multiplication's sign.
        for m in [groups,paths,local,slider]:self.remove(*m.get_family())
        axis=NumberLine(x_range=[-1,1,.5],length=8,include_numbers=True,font_size=22).move_to([0,.3,0])
        q=ValueTracker(.8);prod=always_redraw(lambda:Arrow(axis.n2p(0),axis.n2p(-.6*q.get_value()),buff=0,color=RED_CLASS))
        self.add(axis,prod,number('z_i=',q.get_value,(-2,1.3,0),BLUE_CLASS),tex(r'\delta_j=-0.6',28,RED_CLASS).move_to([2,1.3,0]))
        self.beat(actions=[lambda:q.animate.set_value(-.8),lambda:q.animate.set_value(1)])

    def output_delta(self):
        u=ValueTracker(.1);t=.7
        ax=self.plot_axes(x=(-.6,2,.5),y=(0,.9,.3),width=8.5,height=3.4,labels=('y_k','E_n'))
        graph=curve(ax,lambda y:.5*(y-t)**2,color=BLUE_CLASS)
        dot=always_redraw(lambda:Dot(ax.c2p(u.get_value(),.5*(u.get_value()-t)**2),color=YELLOW_ACC))
        tan=always_redraw(lambda:tangent(ax,u.get_value(),.5*(u.get_value()-t)**2,u.get_value()-t,half=.23))
        self.add(graph,dot,tan,number(r'\delta_k=',lambda:u.get_value()-t,(4.7,1,0),RED_CLASS),knob('y_k=',u,-.6,2,at=(-1,-2.5,0)))
        self.equation(r'y_k=a_k,\qquad E_n=\frac12\sum_k(y_k-t_k)^2')
        target=Dot(ax.c2p(t,0),color=GREEN_CLASS);self.add(target,tex('t_k=0.7',23,GREEN_CLASS).next_to(target,UP,buff=.2))
        self.beat(actions=[lambda:Indicate(target),lambda:ShowPassingFlash(graph.copy(),time_width=.5)])
        self.beat(actions=[lambda:u.animate.set_value(1.7),lambda:u.animate.set_value(1.4)])
        self.beat(actions=[lambda:u.animate.set_value(-.3),lambda:u.animate.set_value(.1)])
        self.equation(r'\delta_k=\frac{\partial E_n}{\partial a_k}=\frac12\cdot2(y_k-t_k)=y_k-t_k')
        self.beat(actions=[lambda:Indicate(self.formula),lambda:u.animate.set_value(t)])
        self.cancellation_recap()
        self.add(note('図：線形出力＋二乗和誤差',at=(3.5,1.85,0)))
        self.equation(r'\sigma+\mathrm{binary\ CE}\quad;\quad\mathrm{softmax}+\mathrm{multiclass\ CE}\quad\Rightarrow\delta_k=y_k-t_k',size=27)
        glossary=VGroup(jp('σ：一つの確率',19),jp('softmax：確率の和が１',19),jp('CE：確率予測の誤差',19)).arrange(DOWN,buff=.2).move_to([5,-.5,0])
        self.add(glossary)
        self.beat(actions=[lambda:Indicate(self.formula),lambda:u.animate.set_value(1.1)])
        self.equation(r'\delta_k=\sum_l\frac{\partial E_n}{\partial y_l}\frac{\partial y_l}{\partial a_k}',size=37)
        self.beat(actions=[lambda:Indicate(self.formula),lambda:u.animate.set_value(.7)])

    def branch(self):
        z=float(np.tanh(.5)); t2=ValueTracker(-.1);v=np.array([1.1,-.6]);y=v*z+np.array([.05,-.1])
        d=lambda: y-np.array([.3,t2.get_value()]);con=lambda:v*d()
        hidden=Circle(radius=.4,color=GREEN_CLASS).move_to([-3,.7,0])
        outs=VGroup(*[Circle(radius=.38,color=RED_CLASS).move_to([2,1.4-1.5*k,0]) for k in range(2)])
        paths=VGroup(*[Line(hidden.get_center(),o.get_center(),buff=.42,color=MUTED) for o in outs])
        reverse=VGroup(*[Line(o.get_center(),hidden.get_center(),buff=.42) for o in outs])
        self.add(hidden,outs,paths,tex('z_j',28,GREEN_CLASS).move_to(hidden))
        for k,o in enumerate(outs):
            self.add(tex(fr'y_{k+1}',25).move_to(o),tex(fr'w_{{{k+1}j}}={v[k]:.1f}',24,YELLOW_ACC).move_to([-.4,1.65-2.1*k,0]),number(fr'\delta_{k+1}=',lambda k=k:d()[k],(4.5,1.4-1.5*k,0),RED_CLASS,size=25))
        line=NumberLine(x_range=[-1,1.5,.5],length=9,include_numbers=True,font_size=22).move_to([-.5,-1.6,0])
        self.add(line)
        self.equation(r'z_j\ \longrightarrow\ (y_1,y_2)\ \longrightarrow\ E_n')
        self.beat(actions=[lambda:pulse(paths,BLUE_CLASS),lambda:Indicate(outs)])
        a1=always_redraw(lambda:Arrow(line.n2p(0)+UP*.2,line.n2p(con()[0])+UP*.2,buff=0,color=RED_CLASS))
        self.equation(r'w_{1j}\delta_1',size=37);self.add(a1)
        self.beat(actions=[lambda:pulse(reverse[:1]),lambda:Circumscribe(a1)])
        a2=always_redraw(lambda:Arrow(line.n2p(0)+DOWN*.6,line.n2p(con()[1])+DOWN*.6,buff=0,color=PURPLE_ACC))
        self.equation(r'w_{2j}\delta_2',size=37);self.add(a2)
        self.beat(actions=[lambda:pulse(reverse[1:],PURPLE_ACC),lambda:Circumscribe(a2)])
        self.remove(a2)
        a2=always_redraw(lambda:Arrow(line.n2p(con()[0])+UP*.2,line.n2p(sum(con()))+UP*.2,buff=0,color=PURPLE_ACC))
        totaldot=always_redraw(lambda:Dot(line.n2p(sum(con())),color=YELLOW_ACC))
        self.equation(r'\frac{\partial E_n}{\partial z_j}=\sum_k w_{kj}\delta_k',size=37)
        self.add(a2,totaldot)
        self.beat(actions=[lambda:Circumscribe(a2),lambda:Circumscribe(totaldot)])
        self.add(number('t_2=',t2.get_value,(-3,-2.5,0),YELLOW_ACC),number(r'\sum w\delta=',lambda:sum(con()),(2,-2.5,0),YELLOW_ACC))
        self.beat(actions=[lambda:t2.animate.set_value(-1),lambda:t2.animate.set_value(-.1)])
        self.equation(r'\delta_j=',r"h'(a_j)",r'\sum_k w_{kj}\delta_k',size=40)
        self.formula[1].set_color(GREEN_CLASS);self.formula[2].set_color(RED_CLASS)
        self.beat(actions=[lambda:Indicate(hidden),lambda:pulse(reverse)])

    def activation(self):
        a=ValueTracker(-.4)
        ax=self.plot_axes(x=(-3,3,1),y=(-1,1,.5),width=8.4,height=3.3,labels=('a_j','z_j'))
        graph=curve(ax,np.tanh,color=GREEN_CLASS)
        dot=always_redraw(lambda:Dot(ax.c2p(a.get_value(),np.tanh(a.get_value())),color=YELLOW_ACC))
        tan=always_redraw(lambda:tangent(ax,a.get_value(),np.tanh(a.get_value()),1-np.tanh(a.get_value())**2,half=.5))
        self.add(graph,dot,tan,knob('a_j=',a,-3,3,at=(-1,-2.5,0)),number("h'=",lambda:1-np.tanh(a.get_value())**2,(4.9,1.1,0),RED_CLASS))
        self.equation(r'h(a)=\tanh(a)=\frac{e^a-e^{-a}}{e^a+e^{-a}}')
        self.beat(a.animate.set_value(.4))
        self.beat(a.animate.set_value(2.7))
        self.equation(r"h'(a_j)=1-\tanh^2(a_j)=1-z_j^2",size=37)
        self.beat(a.animate.set_value(.68))
        # Replace the graph with the actual two-layer sensitivities.
        keep=[self.formula,self.subtitle]
        self.remove(*[m for m in self.mobjects if m not in keep and m.get_center()[1]<3])
        b=backward();net,edges,layers=network(lambda:b,deltas=True)
        self.add(net,note('隠れ層の感度（中央）と出力の感度（右）'))
        rev=VGroup(*[Line(p.get_end(),p.get_start()) for p in edges[1]])
        self.equation(r'\delta_j=(1-z_j^2)\sum_k w^{(2)}_{kj}\delta_k',size=37)
        self.beat(actions=[lambda:pulse(rev),lambda:Circumscribe(layers[1])])
        self.remove(net,*[m for m in self.mobjects if m.get_center()[1]<-2 and m is not self.subtitle])
        table=self.gradient_table(b['d2'],b['zb'],b['g2'],r'\delta_k',r'z_j')
        self.equation(r'\frac{\partial E_n}{\partial w^{(2)}_{kj}}=\delta_kz_j',size=38)
        self.add(table)
        self.beat(actions=[lambda:LaggedStart(*[Indicate(c) for c in table[-1]],lag_ratio=.2),lambda:Indicate(table[-1])])
        new=self.gradient_table(b['d1'],b['x'],b['g1'],r'\delta_j',r'x_i')
        self.equation(r'\frac{\partial E_n}{\partial w^{(1)}_{ji}}=\delta_jx_i',size=38)
        self.beat(actions=[lambda:Transform(table,new),lambda:Indicate(table[-1])])

    def gradient_table(self,d,z,g,dl,zl):
        cols=VGroup(*[tex(f'{v:.3f}',29,BLUE_CLASS).move_to([-1.5+2.2*i,1.45,0]) for i,v in enumerate(z)])
        rows=VGroup(*[tex(f'{v:.3f}',29,RED_CLASS).move_to([-3.5,.35-1.3*i,0]) for i,v in enumerate(d)])
        labels=VGroup(tex(zl,28,BLUE_CLASS).move_to([.7,1.95,0]),tex(dl,28,RED_CLASS).move_to([-4.7,-.3,0]))
        entries=VGroup()
        for j in range(2):
            for i in range(3):
                pos=np.array([-1.5+2.2*i,.35-1.3*j,0])
                entries.add(VGroup(Rectangle(width=1.8,height=.9,color=MUTED,stroke_width=1),tex(f'{g[j,i]:.3f}',30,YELLOW_ACC)).move_to(pos))
        return VGroup(cols,rows,labels,entries)

    def dataset(self):
        bs=[backward(x,t) for x,t in DATA];vs=[b['g2'][0,1:] for b in bs];g=sum(vs)
        ax=self.plot_axes(x=(-.15,1.05,.3),y=(-.9,.3,.3),width=8.5,height=3.4,labels=('g_1','g_2'))
        arrows=VGroup(*[Arrow(ax.c2p(0,0),ax.c2p(*v),buff=0,color=c) for v,c in zip(vs,[BLUE_CLASS,GREEN_CLASS,PURPLE_ACC])])
        self.add(arrows,note('二成分の表示：横は出力１←隠れ１、縦は出力１←隠れ２'))
        self.equation(r'\nabla E_n\quad(n=1,2,3)')
        self.beat(actions=[lambda:Indicate(arrows[0]),lambda:LaggedStart(Indicate(arrows[1]),Indicate(arrows[2]),lag_ratio=.5)])
        targets=[];acc=np.zeros(2)
        for v,c in zip(vs,[BLUE_CLASS,GREEN_CLASS,PURPLE_ACC]):
            targets.append(Arrow(ax.c2p(*acc),ax.c2p(*(acc+v)),buff=0,color=c));acc+=v
        self.beat(actions=[lambda:Indicate(arrows),lambda:AnimationGroup(*[Transform(a,b) for a,b in zip(arrows,targets)])])
        totalarrow=Arrow(ax.c2p(0,0),ax.c2p(*g),buff=0,color=YELLOW_ACC)
        self.equation(r'E=\sum_n E_n\Rightarrow\nabla E=\sum_n\nabla E_n')
        self.add(totalarrow)
        self.beat(actions=[lambda:Indicate(totalarrow),lambda:Transform(totalarrow,Arrow(ax.c2p(0,0),ax.c2p(*(g/3)),buff=0,color=YELLOW_ACC))])
        # Actual full 12-parameter update, with the loss recalculated along the step.
        g1,g2=batch();u=ValueTracker(0)
        self.remove(*arrows.get_family(),*totalarrow.get_family())
        self.equation(r'\mathbf w_{\rm new}=\mathbf w-0.1\nabla E')
        for label,symbol in zip(ax.axis_labels,[r'\Delta w_1',r'\Delta w_2']):
            label.become(tex(symbol,25).move_to(label.get_center()))
        move=always_redraw(lambda:Arrow(ax.c2p(0,0),ax.c2p(*(-u.get_value()*g)),buff=0,color=RED_CLASS))
        read=number('E=',lambda:total(W1-u.get_value()*g1,W2-u.get_value()*g2),(4.5,-.7,0),YELLOW_ACC)
        self.add(move,read)
        self.beat(actions=[lambda:u.animate.set_value(.1),lambda:Circumscribe(read)])
        self.remove(*[m for m in self.mobjects if m is not self.formula and m is not self.subtitle and m.get_center()[1]<3])
        self.equation(r'\mathrm{forward}\quad\rightarrow\quad\mathrm{backward}\quad\rightarrow\quad\delta_jz_i')
        net,edges,layers=network(lambda:forward());self.add(net)
        rev=VGroup(*[Line(p.get_end(),p.get_start()) for group in edges[::-1] for p in group])
        self.beat(actions=[lambda:Succession(pulse(VGroup(*edges),BLUE_CLASS),pulse(rev)),lambda:Circumscribe(layers[1])])

    def efficiency(self):
        eps=ValueTracker(.6);w=.2
        ax=self.plot_axes(x=(-.6,1,.4),y=(0,.9,.3),width=8.5,height=3.3,labels=('w','E_n'))
        graph=curve(ax,lambda u:scalar(u)['E'],color=BLUE_CLASS)
        sec=always_redraw(lambda:tangent(ax,w,(scalar(w+eps.get_value())['E']+scalar(w-eps.get_value())['E'])/2,central(w,eps.get_value()),YELLOW_ACC,half=max(.3,eps.get_value())))
        dots=always_redraw(lambda:VGroup(*[Dot(ax.c2p(w+s*eps.get_value(),scalar(w+s*eps.get_value())['E']),color=YELLOW_ACC) for s in [-1,1]]))
        self.add(graph,sec,dots,number(r'\epsilon=',eps.get_value,(-3,-2.5,0),YELLOW_ACC,places=3),number('g_{CD}=',lambda:central(w,eps.get_value()),(2,-2.5,0),YELLOW_ACC,places=4))
        self.equation(r'\frac{E_n(w+\epsilon)-E_n(w-\epsilon)}{2\epsilon}=\frac{\partial E_n}{\partial w}+O(\epsilon^2)',size=33)
        self.beat(actions=[lambda:Circumscribe(dots),lambda:Circumscribe(sec)])
        self.add(tangent(ax,w,scalar(w)['E'],scalar(w)['g'],RED_CLASS,half=.3),tex(fr'g_{{BP}}={scalar(w)["g"]:.4f}',25,RED_CLASS).move_to([4.7,1.1,0]))
        self.beat(eps.animate.set_value(.005))
        self.beat(actions=[lambda:eps.animate.set_value(.001),lambda:Circumscribe(sec)])
        self.remove(*[m for m in self.mobjects if m is not self.formula and m is not self.subtitle and m.get_center()[1]<3])
        # Log coordinates keep both asymptotic laws legible on one set of axes.
        u=ValueTracker(1)
        ax=self.plot_axes(x=(0,3,1),y=(0,7,1),width=8.5,height=3.4,labels=(r'\log_{10}W',r'\log_{10}C'))
        bp=curve(ax,lambda q:q,color=BLUE_CLASS);cd=curve(ax,lambda q:np.log10(2)+2*q,color=RED_CLASS)
        dots=always_redraw(lambda:VGroup(Dot(ax.c2p(u.get_value(),u.get_value()),color=BLUE_CLASS),Dot(ax.c2p(u.get_value(),np.log10(2)+2*u.get_value()),color=RED_CLASS)))
        self.add(jp('復習: 1.1 対数の目盛り',20).move_to([0,2.98,0]))
        self.add(bp,cd,dots,note('計算量の模型：青 C = W ／ 赤 C = 2W²（両軸は常用対数）'))
        self.equation(r'\mathrm{backprop}:O(W)\qquad\mathrm{central\ difference}:O(W^2)',size=31)
        self.beat(u.animate.set_value(2))
        self.beat(u.animate.set_value(3))
        self.beat(actions=[lambda:Indicate(bp),lambda:u.animate.set_value(1.5)])

    def input_jacobian(self):
        u=ValueTracker(0);base=X.copy();direction=np.array([1.,-.5]);y0=forward()['y'];J=jacobian()
        f=lambda s:forward(base+s*direction)['y']
        ax=self.plot_axes(x=(.8,2.15,.3),y=(-.62,-.3,.1),width=7.4,height=3.3,center=(-1,-.15,0),labels=('y_1','y_2'))
        # Suppress the first x tick number at the upper-left axis intersection.
        ax.x_axis.numbers[0].set_opacity(0)
        samples=np.linspace(-.2,1.4,150)
        path=VMobject().set_points_as_corners([ax.c2p(*f(s)) for s in samples]).set_stroke(BLUE_CLASS,3)
        dot=always_redraw(lambda:Dot(ax.c2p(*f(u.get_value())),color=BLUE_CLASS,radius=.085))
        pred=always_redraw(lambda:Dot(ax.c2p(*(y0+u.get_value()*(J@direction))),color=YELLOW_ACC,radius=.065))
        self.add(path,dot,knob('s=',u,0,1.2,at=(-1,-2.5,0)))
        self.equation(r'\mathbf x=\mathbf x_0+s(1,-0.5)^T,\quad\mathbf w\ \mathrm{fixed}')
        self.beat(actions=[lambda:u.animate.set_value(.15),lambda:u.animate.set_value(0)])
        self.equation(r'J_{ki}=\frac{\partial y_k}{\partial x_i}',size=39)
        matrix=Matrix([[f'{v:.3f}' for v in row] for row in J],h_buff=1.8,v_buff=.75).scale(.52).move_to([4.8,1.25,0])
        self.add(matrix)
        self.beat(actions=[lambda:Indicate(matrix.get_rows()[0]),lambda:Indicate(matrix.get_rows()[1])])
        self.jacobian_columns_aid()
        self.equation(r'\Delta\mathbf y\simeq J(\mathbf x_0)\Delta\mathbf x',size=37)
        self.add(pred,note('青：実際の出力　黄色：基準点での線形近似',at=(0,-2.05,0)))
        self.beat(actions=[lambda:u.animate.set_value(.2),lambda:u.animate.set_value(.05)])
        self.beat(actions=[lambda:u.animate.set_value(1.1),lambda:u.animate.set_value(.15)])
        self.remove(*[m for m in self.mobjects if m is not self.formula and m is not self.subtitle and m.get_center()[1]<3])
        self.equation(r'\frac{\partial y_k}{\partial a_j}=h^{\prime}(a_j)\sum_lw_{lj}\frac{\partial y_k}{\partial a_l}',size=33)
        net,edges,layers=network(lambda:forward());self.add(net)
        rev=VGroup(*[Line(p.get_end(),p.get_start()) for group in edges[::-1] for p in group])
        bottom=tex(r'J_{ki}=\sum_j w_{ji}\frac{\partial y_k}{\partial a_j}',31,YELLOW_ACC).move_to([0,-2.1,0]);self.add(bottom)
        self.beat(actions=[lambda:pulse(rev),lambda:Indicate(bottom)])
        self.remove(net,bottom)
        seeds=VGroup(
            tex(r'\mathrm{sigmoid}:\quad\frac{\partial y_k}{\partial a_j}=\delta_{kj}\sigma^{\prime}(a_j)',34,BLUE_CLASS),
            tex(r'\mathrm{softmax}:\quad\frac{\partial y_k}{\partial a_j}=\delta_{kj}y_k-y_ky_j',34,RED_CLASS),
            tex(r'\delta_{kj}=1\ (k=j),\quad0\ (k\ne j)',28,MUTED)).arrange(DOWN,buff=.6).move_to([0,-.1,0])
        self.equation(r'\text{output seeds}\quad\longrightarrow\quad J_{k,:}')
        self.add(seeds)
        self.beat(actions=[lambda:Indicate(seeds[0]),lambda:Indicate(seeds[1])])
        self.remove(seeds)
        blocks=VGroup(*[RoundedRectangle(width=2,height=1,corner_radius=.15,color=c).move_to(p) for p,c in [([-2,1,0],GREEN_CLASS),([-2,-1,0],BLUE_CLASS),([2,0,0],RED_CLASS)]])
        arrows=VGroup(Line(blocks[0].get_right(),blocks[2].get_left()),Line(blocks[1].get_right(),blocks[2].get_left()),Arrow([3,0,0],[5,0,0],buff=0))
        self.add(blocks,arrows,tex('u',28).move_to([-4,1,0]),tex('x',28).move_to([-4,-1,0]),tex('y',28).move_to([5.4,0,0]),tex('v',25).move_to([.1,1,0]),tex('z',25).move_to([.1,-1,0]),tex('w',28,YELLOW_ACC).move_to(blocks[1]))
        self.add(Arrow([-3.7,1,0],[-3,1,0],buff=0),Arrow([-3.7,-1,0],[-3,-1,0],buff=0))
        self.equation(r'\frac{\partial E}{\partial w}=\sum_{k,j}\frac{\partial E}{\partial y_k}\frac{\partial y_k}{\partial z_j}\frac{\partial z_j}{\partial w}',size=34)
        self.beat(actions=[lambda:pulse(arrows,BLUE_CLASS),lambda:pulse(VGroup(*[Line(p.get_end(),p.get_start()) for p in arrows[::-1]]))])

    def body_card(self,label):
        """Replace the body for one PCM-timed beat, preserving the title."""
        saved=[m for m in self.mobjects if m is not self.subtitle]
        self.clear()
        self.add(*[m for m in saved if m.get_center()[1]>3.1])
        self.add(RoundedRectangle(width=10.4,height=4.6,corner_radius=.12,
                                  color='#FFFF00',stroke_width=1.2).move_to([0,.1,0]),
                 jp(label,23).move_to([-4.85,2.06,0],aligned_edge=LEFT))
        return saved

    def restore_body(self,saved):
        self.clear();self.add(*saved);self.subtitle=None

    def cancellation_recap(self):
        saved=self.body_card('復習: 4.3 シグモイドと交差エントロピー')
        # 4.3 cross_entropy: yellow residual, green feature; source palette.
        nodes=VGroup(*[VGroup(RoundedRectangle(width=1.7,height=.8,
                         corner_radius=.08,color=c),tex(label,32,c)).move_to([x,.85,0])
                       for x,label,c in [(-3.7,'a',BLUE_CLASS),(0,'y',YELLOW_ACC),(3.7,'E_n',RED_CLASS)]])
        paths=VGroup(Arrow([-2.8,.85,0],[-.9,.85,0],buff=.05,color=MUTED),
                     Arrow([.9,.85,0],[2.8,.85,0],buff=.05,color=MUTED))
        self.add(nodes,paths,tex(r'y=\sigma(a),\quad t\in\{0,1\},\quad 0<y<1',26).move_to([0,1.5,0]),
                 jp('シグモイド',19).move_to([-1.85,.32,0]),
                 jp('交差エントロピー',19).move_to([1.95,.32,0]))
        numerator=tex('y-t',34,YELLOW_ACC).move_to([-1.75,-.4,0])
        denominator=tex('y(1-y)',30).move_to([-1.75,-1.08,0])
        rule=Line([-2.5,-.72,0],[-1,-.72,0],color=WHITE,stroke_width=1.5)
        factor=tex('y(1-y)',30).move_to([1.15,-.72,0])
        times=tex(r'\cdot',34).move_to([-.25,-.72,0])
        left=tex(r'\frac{\partial E_n}{\partial a}=',31).move_to([-3.8,-.72,0])
        factors=VGroup(numerator,denominator,rule,times,factor)
        crosses=VGroup(*[Line(m.get_corner(DL),m.get_corner(UR),color=YELLOW_ACC,stroke_width=3)
                          for m in [denominator,factor]])
        result=tex('y-t',38,YELLOW_ACC).move_to([-1.75,-.72,0])
        mapping=VGroup(tex('y-t',32,YELLOW_ACC),Arrow(LEFT*.3,RIGHT*.3,buff=0,color=MUTED),
                       tex(r'\delta_k=y_k-t_k',32,RED_CLASS)).arrange(RIGHT,buff=.23).move_to([0,-1.78,0])
        self.add(left,factors)
        def cancel():
            return Succession(Create(crosses),AnimationGroup(FadeOut(crosses),ReplacementTransform(factors,result)))
        self.beat(actions=[lambda:pulse(paths,YELLOW_ACC),cancel,lambda:FadeIn(mapping,shift=DOWN*.1)])
        self.restore_body(saved)

    def jacobian_columns_aid(self):
        saved=self.body_card('補足: 一つの入力を動かすと、行列の一列へ')
        blue,yellow,green='#58C4DD','#FFFF00','#83C167'
        J=jacobian();y0=forward()['y'];column=[0];eps=ValueTracker(.0005)
        displacement=lambda:np.eye(2)[column[0]]*eps.get_value()
        change=lambda:forward(X+displacement())['y']-y0
        self.add(tex(r'\mathbf x_0=(0.6,-0.4)^T',25).move_to([-1.2,1.48,0]),
                 number(r'\epsilon=',eps.get_value,[2.4,1.48,0],blue,places=4,size=25),
                 jp('他の入力・重みは固定',19).move_to([-3.1,-1.8,0]))
        inputs=always_redraw(lambda:VGroup(tex(r'\Delta\mathbf x=',27,blue),
                    Matrix([[f'{v:.4f}'] for v in displacement()],v_buff=.8).scale(.55).set_color(blue)
                    ).arrange(RIGHT,buff=.15).move_to([-3.4,.1,0]))
        table=Matrix([[f'{v:.3f}' for v in row] for row in J],h_buff=1.8,v_buff=1.3).scale(.6).set_color(green).move_to([3.6,.05,0])
        columns=table.get_columns()
        label=tex('J=',30,green).next_to(table,LEFT,buff=.2)
        self.add(inputs,table,label,
                 tex(r'x_1\qquad x_2',25,blue).move_to([3.6,1.03,0]),
                 tex(r'\Delta\mathbf y/\epsilon\approx J_{:,i}',27,green).move_to([2.3,-1.8,0]))
        self.add(*[tex(r'\div\epsilon\;\longrightarrow',23,green).move_to([1.35,y,0]) for y in [.42,-.73]])
        current=[]
        def show_column(i):
            self.remove(*current);current.clear();column[0]=i;eps.set_value(.0005)
            # Separate scales expose the small y2 response without changing values.
            for k in range(2):
                span=max(abs(J[k,i])*.027,.0004)
                line=NumberLine(x_range=[-span,span,span],length=2.0,include_numbers=False,
                                color=MUTED).move_to([-.4,.42-1.15*k,0])
                arrow=always_redraw(lambda k=k,line=line:Arrow(line.n2p(0),line.n2p(change()[k]),
                                     buff=0,color=yellow,stroke_width=4,max_tip_length_to_length_ratio=.3))
                value=number(fr'\Delta y_{k+1}=',lambda k=k:change()[k],[-.45,.82-1.15*k,0],yellow,places=5,size=23)
                current.extend([line,arrow,value])
            highlight=SurroundingRectangle(columns[i],color=yellow,buff=.13)
            fixed=tex(fr'\Delta x_{{{2-i}}}=0',25,blue).move_to([-3.4,-1.03,0])
            current.extend([highlight,fixed]);self.add(*current)
            return Succession(eps.animate.set_value(.02),
                AnimationGroup(*[TransformFromCopy(current[3*k+2][1],columns[i][k]) for k in range(2)]))
        self.add(jp('矢印の尺度は各出力で拡大',17,MUTED).move_to([-.4,-1.4,0]))
        self.beat(actions=[lambda:show_column(0),lambda:show_column(1)])
        self.restore_body(saved)
