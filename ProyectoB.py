#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, random, argparse, os, sys, time, csv, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- distancia geodésica ----------
R = 6371.0088
def hav(l1, L1, l2, L2):
    φ1, φ2 = map(math.radians, (l1, l2))
    dφ, dλ = φ2-φ1, math.radians(L2-L1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2*R*math.asin(math.sqrt(a))

# ---------- lectura de CSV ----------
def load_data(folder: str = '.') -> tuple:
    """
    Lee depots.csv, clients.csv y vehicles.csv.
    • Si `folder` es '' o '.' usa la carpeta donde reside este .py.
    • Devuelve: matriz de distancias, lista de demandas, lista de coordenadas, nº vehículos.
    """
    if folder in ('', '.'):
        folder = os.path.dirname(os.path.abspath(__file__))

    try:
        ddf = pd.read_csv(os.path.join(folder, 'depots.csv'))
        cdf = pd.read_csv(os.path.join(folder, 'clients.csv'))
        vdf = pd.read_csv(os.path.join(folder, 'vehicles.csv'))
    except FileNotFoundError as e:
        print('CSV no encontrado →', e)
        sys.exit(1)

    # depósito único (fila 0)
    latd = ddf.filter(regex='(?i)Lat').iloc[0].values[0]
    lond = ddf.filter(regex='(?i)Lon').iloc[0].values[0]
    coords = [(latd, lond)]

    # clientes
    latc = cdf.filter(regex='(?i)Lat').columns[0]
    lonc = cdf.filter(regex='(?i)Lon').columns[0]
    demc = cdf.filter(regex='(?i)Demand').columns[0]
    coords += list(zip(cdf[latc].astype(float), cdf[lonc].astype(float)))
    demands = cdf[demc].astype(float).tolist()

    # matriz de distancias geodésicas
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = hav(*coords[i], *coords[j])

    # número de vehículos (filas en vehicles.csv) – capacity homogénea ya se pasa por --Q
    nveh = max(1, len(vdf))

    return D, demands, coords, nveh


# ---------- GA CVRP ----------
class GA:
    def __init__(self, dist, dem, Q, nv, pop, gen, mut, crx, elite, log):
        self.D, self.dem, self.Q, self.nv = dist, dem, Q, nv
        self.cli = list(range(1, len(dem)+1))
        self.P, self.G, self.MUT, self.CRX = pop, gen, mut, crx
        self.EL = max(1, int(elite*pop)); self.log = max(1, log)
        self.pop, self.best, self.bestfit, self.hist = [], None, 1e99, []

    def _rnd(self):
        routes=[[] for _ in range(self.nv)]
        bag=self.cli.copy(); random.shuffle(bag)
        for c in bag:
            v=min(range(self.nv), key=lambda i: sum(self.dem[k-1] for k in routes[i]))
            routes[v].append(c)
        return self._repair(routes)

    def _repair(self, sol):
        seen=set(); miss=[]
        for r in sol:
            r[:]=[c for c in r if not (c in seen or seen.add(c))]
        miss=[c for c in self.cli if c not in seen]
        for c in miss: sol[random.randrange(self.nv)].append(c)
        new=[]
        for r in sol:
            load=0; buf=[]
            for c in r:
                if load+self.dem[c-1]>self.Q:
                    new.append(buf); buf=[]; load=0
                buf.append(c); load+=self.dem[c-1]
            if buf: new.append(buf)
        while len(new)<self.nv: new.append([])
        return new

    def _rcost(self, r):
        if not r: return 0
        c=self.D[0,r[0]]+self.D[r[-1],0]
        for i in range(len(r)-1): c+=self.D[r[i],r[i+1]]
        return c
    def _fit(self, s): return sum(self._rcost(r) for r in s)
    def _tour(self):
        a,b=random.sample(self.pop,2)
        return a if self._fit(a)<self._fit(b) else b
    def _cross(self,p1,p2):
        if random.random()>self.CRX: return [p1[:],p2[:]]
        c1,c2=[r[:] for r in p1],[r[:] for r in p2]
        i=random.randrange(len(c1)); j=random.randrange(len(c2))
        c1[i],c2[j]=c2[j],c1[i]
        return self._repair(c1),self._repair(c2)
    def _mut(self,s):
        if random.random()>self.MUT: return s
        r1=[r[:] for r in s]
        a,b=random.sample(self.cli,2)
        loc1=loc2=None
        for r in r1:
            if a in r: loc1=(r,a); 
            if b in r: loc2=(r,b)
        if loc1 and loc2:
            loc1[0][loc1[0].index(a)]=b
            loc2[0][loc2[0].index(b)]=a
        return self._repair(r1)

    def run(self):
        self.pop=[self._rnd() for _ in range(self.P)]
        tic=time.time()
        for g in range(self.G):
            fits=[self._fit(s) for s in self.pop]; idx=int(np.argmin(fits))
            if fits[idx]<self.bestfit: self.bestfit=fits[idx]; self.best=[r[:] for r in self.pop[idx]]
            self.hist.append(self.bestfit)
            if g%self.log==0 or g==self.G-1:
                print(f'Gen {g:3d} | best {self.bestfit:.3f} | t {time.time()-tic:.1f}s',flush=True)
            elite=[self.pop[i] for i in np.argsort(fits)[:self.EL]]
            new=elite[:]
            while len(new)<self.P:
                c1,c2=self._cross(self._tour(),self._tour())
                new.append(self._mut(c1))
                if len(new)<self.P: new.append(self._mut(c2))
            self.pop=new
        return self.best,self.bestfit

# ---------- gráficos ----------
def g_convergence(hist, outfile):
    plt.figure(); plt.plot(hist); plt.grid(); plt.xlabel('Gen'); plt.ylabel('Best'); plt.tight_layout(); plt.savefig(outfile); plt.close()
def g_routes(best, coords, outpng):
    xs=[c[1] for c in coords]; ys=[c[0] for c in coords]
    plt.figure(); plt.scatter(xs[0],ys[0],c='red')
    plt.scatter(xs[1:],ys[1:],c='black',s=10)
    for k,r in enumerate(best):
        seq=[0]+r+[0]
        plt.plot([xs[i] for i in seq],[ys[i] for i in seq],'-o',markersize=3,label=f'R{k+1}')
    plt.legend(); plt.tight_layout(); plt.savefig(outpng); plt.close()
def g_histograms(best, dem, outpng):
    loads=[sum(dem[c-1] for c in r) for r in best if r]
    lens =[len(r) for r in best if r]
    fig,(ax1,ax2)=plt.subplots(1,2, figsize=(6,3))
    ax1.boxplot(loads); ax1.set_title('Carga'); ax2.boxplot(lens); ax2.set_title('Clientes'); plt.tight_layout(); plt.savefig(outpng); plt.close()

# ---------- verificación csv ----------
def write_verif(best, dist, case, folder):
    fn=f'verificacion_metaheuristica_GA_{case}.csv'
    path=os.path.join(folder,fn)
    with open(path,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['Route','Sequence','Distance'])
        for k,r in enumerate(best):
            if not r: continue
            seq=[0]+r+[0]
            d=sum(dist[seq[i],seq[i+1]] for i in range(len(seq)-1))
            w.writerow([k+1,'-'.join(map(str,seq)),f'{d:.3f}'])

# ---------- main ----------
if __name__=='__main__':
    ap=argparse.ArgumentParser('GA CVRP')
    ap.add_argument('-d','--dir',default='.',help='folder instancia')
    ap.add_argument('--Q',type=float,default=120,help='capacidad vehículo')
    ap.add_argument('--pop',type=int,default=120)
    ap.add_argument('--gen',type=int,default=400)
    ap.add_argument('--mut',type=float,default=0.2)
    ap.add_argument('--crx',type=float,default=0.8)
    ap.add_argument('--elite',type=float,default=0.1)
    ap.add_argument('--log',type=int,default=50)
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--case',default='X')
    args=ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed)

    D, Dem, Coords, nveh = load_data(args.dir)
    ga=GA(D, Dem, args.Q, nveh, args.pop, args.gen, args.mut, args.crx, args.elite, args.log)
    best, best_cost = ga.run()

    pathlib.Path(args.dir).mkdir(exist_ok=True, parents=True)
    g_convergence(ga.hist, os.path.join(args.dir,'convergencia.png'))
    g_routes(best, Coords, os.path.join(args.dir,'rutas.png'))
    g_histograms(best, Dem, os.path.join(args.dir,'cargas_clientes.png'))
    write_verif(best, D, args.case, args.dir)
