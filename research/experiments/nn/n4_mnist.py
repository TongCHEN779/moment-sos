"""N4: certified robustness of a small MNIST classifier (digits 3 vs 8, PCA inputs): IBP vs order 1 vs order 2 vs exact MILP.
Usage: python n4_mnist.py MNIST_ROOT [d] [width]"""
from common import *
import time, json, sys
from torchvision.datasets import MNIST
root = sys.argv[1]; d = int(sys.argv[2]) if len(sys.argv) > 2 else 4; wd = int(sys.argv[3]) if len(sys.argv) > 3 else 8
tr, te = MNIST(root, train=True), MNIST(root, train=False)
def sel(ds):
    m = (ds.targets == 3) | (ds.targets == 8)
    return ds.data[m].reshape(-1, 784).double().numpy() / 255.0, (ds.targets[m] == 8).long().numpy()
Xtr, ytr = sel(tr); Xte, yte = sel(te)
mu = Xtr.mean(0); U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); P = Vt[:d].T
Ztr, Zte = (Xtr - mu) @ P, (Xte - mu) @ P
sc = np.abs(Ztr).max(0); Ztr, Zte = np.clip(Ztr / sc, -1, 1), np.clip(Zte / sc, -1, 1)
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(d, wd), torch.nn.ReLU(), torch.nn.Linear(wd, wd), torch.nn.ReLU(), torch.nn.Linear(wd, 2)).double()
opt = torch.optim.Adam(net.parameters(), lr=3e-3); X = torch.as_tensor(Ztr); y = torch.as_tensor(ytr)
for s in range(3000):
    opt.zero_grad(); torch.nn.functional.cross_entropy(net(X), y).backward(); opt.step()
acc = (net(torch.as_tensor(Zte)).argmax(1).numpy() == yte).mean(); print("test acc", acc, flush=True)
lin = [l for l in net if isinstance(l, torch.nn.Linear)]
Ws = [l.weight.detach().numpy() for l in lin]; bs = [l.bias.detach().numpy() for l in lin]
rng = np.random.default_rng(0); out = []
idx = [i for i in rng.permutation(len(yte)) if pnn.forward(Ws, bs, Zte[i]).argmax() == yte[i]][:8]
for i in idx:
    x0 = Zte[i]; cls = int(yte[i]); c = np.zeros(2); c[1 - cls] = 1; c[cls] = -1
    margin = float(abs(np.diff(pnn.forward(Ws, bs, x0))[0]))
    for eps in (0.05, 0.1, 0.2):
        Pb, info = pnn.relu_verification(Ws, bs, x0, eps, c)
        rec = dict(i=int(i), eps=eps, margin=margin, n=info["n"], n_unstable=info["n_unstable"], ibp=info["ibp_upper"],
                   exact=pnn.verification_exact_milp(Ws, bs, x0, eps, c)[0], pgd=pnn.verification_pgd(Ws, bs, x0, eps, c))
        if info["n"] > 0:
            for o in (1, 2):
                try:
                    v, t = exact_relaxation(Pb, o); rec[f"mosek{o}"], rec[f"t_mosek{o}"] = v, t
                    t0 = time.time(); r = po.solve(Pb, order=o, iters_per_stage=4000, etas=(1e-3, 1e-4, 1e-5))
                    rec[f"ours{o}"], rec[f"t_ours{o}"] = r.bound, time.time() - t0
                except Exception as e: rec[f"err{o}"] = str(e)[:80]
        out.append(rec); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        json.dump(dict(acc=float(acc), d=d, width=wd, rows=out), open(f"../results/nn_mnist_d{d}_w{wd}.json", "w"), indent=1)
