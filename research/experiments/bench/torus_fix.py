import sys, os, time
os.chdir("/Users/tchen/Documents/Python/moment-sos/experiments/bench")
sys.path.insert(0, os.getcwd()); __file__ = os.path.join(os.getcwd(), "bm_cert.py")
exec(open("bm_cert.py").read().split('if __name__')[0])
from msos import solvers
seed = int(sys.argv[1]); ips = int(sys.argv[2])
n, edges, w = instance("torus", 36, seed)
pop = pb.maxcut(n, edges, w); d = SpectralDual(pop, 2)
LB = local_search_cut(n, edges, w)
t = time.perf_counter()
U, th, h = solvers.lbfgs_continuation(d, etas=(1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6), iters_per_stage=ips, time_limit=900)
ok = np.where(np.floor(h[:, 3] + 1e-9) <= LB + 1e-9)[0]
print(f"seed {seed} iters/stage {ips}: LB {LB:.0f} final U {U:.3f} (gap {U-LB:.2f}); certified after {h[ok[0],1] if len(ok) else None}s ({int(h[ok[0],0]) if len(ok) else None} evals); total {time.perf_counter()-t:.0f}s", flush=True)
