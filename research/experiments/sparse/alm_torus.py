import sys,os,time,json
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
sys.path.insert(0,os.path.join(ROOT,"..","src")); sys.path.insert(0,ROOT)
import moment_sos as po
f=sys.argv[1]; tl=float(sys.argv[2]); solver=sys.argv[3]
L=open(f).read().split("\n"); n,m=map(int,L[0].split()); E=[];W=[]
for l in L[1:m+1]:
    i,j,w=l.split(); E.append((int(i)-1,int(j)-1)); W.append(float(w))
s=po.variables(n); cut=sum(w*(1-s[i]*s[j])/2 for (i,j),w in zip(E,W)); P=po.Problem(cut,boolean=True)
t=time.time(); r=po.solve_alm(P,order=2,cliques="auto",time_limit=tl,inner=(40 if solver=="newton" else 600),inner_solver=solver,warm_start=True,verbose=True)
print("RESULT",json.dumps(dict(solver=solver,U2=float(r.bound),t=time.time()-t)),flush=True)
