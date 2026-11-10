from common import *
import time
d,widths,seed=4,[8,8],5
Ws,bs,acc=train_mlp(d,widths,seed=seed); rng=np.random.default_rng(100+seed)
for t in range(200):
    x0=rng.uniform(-0.8,0.8,d); lg=pnn.forward(Ws,bs,x0)
    if abs(lg[0]-lg[1])>0.4: break
cls=int(lg.argmax()); c=np.zeros(2); c[1-cls]=1; c[cls]=-1
P,info=pnn.relu_verification(Ws,bs,x0,0.2,c); print('n',info['n'])
ref,_=exact_relaxation(P,2); print('ipm',ref)
for name,kw in (('lbfgs',dict(inner=600)),('newton',dict(inner=40,inner_solver='newton'))):
    t0=time.time(); r=po.solve_alm(P,order=2,outer=60,time_limit=300,**kw)
    print(name,'bound',prob_val:=P.f_off+P.f_scale*r.bound if False else r.bound,'time',round(time.time()-t0,1),'evals',r.n_eval if hasattr(r,'n_eval') else '')
