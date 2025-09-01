import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import itertools
import scipy 
import math
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
pi = torch.tensor(np.pi,dtype=torch.float64)

from utils import PiecewiseGQ2D_weights_points, SquareMesh2D_points, split_domain_boundary
from utils import s2_uniform_grid_init
from utils import output_convergence_order_l2
from utils import plot_err_convergence

class ShallowReLUkFitter:
    def __init__(self, ws, bs, train_pts, k=1, m1=1, m2=1):
        self.n = ws.shape[0]
        self.W = ws 
        self.b = bs
        self.k = k
        self.m1 = m1 
        self.m2 = m2
        self.pts = train_pts 
    
    def feat_extract(self, x):
        return F.relu(x @ self.W.t() + self.b) ** self.k

    def target(self, x):
        z = torch.sin(self.m1 * pi*x[:,0:1])*torch.sin(self.m2 * pi*x[:,1:2])
        return z 

    def assemble(self):
        A = self.feat_extract(self.pts)
        b = self.target(self.pts)
        return A, b

    def solve(self):
        A, b = self.assemble()

        c = 100.0 
        for i in range(len(A)):
            max_a = abs(A[i,:]).max()
            max_b = A[i,:].max()
            if max_a != max_b: 
                ratio = -c/max_a
                A[i,:] = A[i,:]*ratio
                b[i] = b[i]*ratio
            else: 
                ratio = c/max_a
                A[i,:] = A[i,:]*ratio
                b[i] = b[i]*ratio
        alpha = scipy.linalg.lstsq(A.numpy(),b.numpy())[0]
        self.alpha = torch.tensor(alpha)
        
    def forward(self, x):
        return self.feat_extract(x) @ self.alpha
    
    def eval(self, x):
        y_ref = self.target(x)
        y_pred = self.forward(x)
        rel_errl2 = (y_pred - y_ref).norm() / y_ref.norm()
        return rel_errl2.item()

class PouShallowReLUkFitter:
    def __init__(
        self, ws, bs, pts, k=1, l=1, sigma=0.01, m1=1, m2=1):
        
        self.n = ws.shape[0]
        self.d = ws.shape[1]
        self.k = k
        self.l = l
        self.sigma = sigma
        self.m1 = m1 
        self.m2 = m2
        self.pts = pts
        cntrs, rng = self.pou_init()
        self.cntrs, self.rng = cntrs, rng

        self.J = self.cntrs.shape[0]
        self.W = ws.unsqueeze(0).repeat(self.J, 1, 1)   # [J, n, d]
        self.b = bs.unsqueeze(0).repeat(self.J, 1)      # [J, n]
    
    def map_unit(self, x):
        '''
        Inputs:
            x : (N, d)
            cntrs: (J, d)
            rng : float
        Returns:
            x_ : (N, J, d)
        '''
        x_ = (x.unsqueeze(0) - self.cntrs.unsqueeze(1) + self.rng)/(2 * self.rng)
        return x_
    
    def pou_init(self):
        rng = 1 / 2**(self.l)
        cntr_1d = torch.linspace(0, 1, 2**(self.l-1)+1).reshape(-1, 1)
        grid = list(itertools.product(cntr_1d, repeat=2)) # 2 for 2D input
        cntrs = torch.tensor(grid)
        return cntrs, rng

    def pou_extract(self, x):
        '''
        Inputs:
            x : (N, D) tensor
            cntrs : (J, D) tensor 
            rng : scalar (float or tensor)
            sigma : scalar (float or tensor)
        Returns:
            (J, N) tensor
        '''
        def phi(z):
            return 1 / (1 + torch.exp(-z))

        # Expand dimensions for broadcasting: (N,1,D) - (1,M,D) → (N,M,D)
        a = self.cntrs.unsqueeze(1) - self.rng
        b = self.cntrs.unsqueeze(1) + self.rng
        x_exp = x.unsqueeze(0)

        # Apply PoU formula over each dimension
        vals = phi((x_exp - a) / self.sigma) * phi((b - x_exp) / self.sigma)

        # Product over D → (N,M)
        return vals.prod(dim=2, keepdim=False)

    def feat_extract(self, x):
        pou_feat = self.pou_extract(x) # [J, N]
        x_ = self.map_unit(x)
        local_feats = []
        for j in range(self.J):
            feat = F.relu(x_[j] @ self.W[j].t() + self.b[j]) ** self.k
            local_feat = feat * pou_feat[j].unsqueeze(-1)
            local_feats.append(local_feat)
        return torch.hstack(local_feats)

    def target(self, x):
        z = torch.sin(self.m1 * pi*x[:,0:1])*torch.sin(self.m2 * pi*x[:,1:2])
        return z 

    def assemble(self):
        A = self.feat_extract(self.pts)
        b = self.target(self.pts)
        return A, b

    def solve(self):
        A, b = self.assemble()

        c = 100.0 
        for i in range(len(A)):
            max_a = abs(A[i,:]).max()
            max_b = A[i,:].max()
            if max_a != max_b: 
                ratio = -c/max_a
                A[i,:] = A[i,:]*ratio
                b[i] = b[i]*ratio
            else: 
                ratio = c/max_a
                A[i,:] = A[i,:]*ratio
                b[i] = b[i]*ratio
        alpha = scipy.linalg.lstsq(A.numpy(),b.numpy())[0]
        self.alpha = torch.tensor(alpha)
        
    def forward(self, x):
        return self.feat_extract(x) @ self.alpha
    
    def eval(self, x):
        y_ref = self.target(x)
        y_pred = self.forward(x)
        rel_errl2 = (y_pred - y_ref).norm() / y_ref.norm()
        return rel_errl2.item()

class ShallowReLUkSolver:
    def __init__(self, ws, bs, train_pts, k=1, m1=1, m2=1):
        self.n = ws.shape[0]
        self.W = ws 
        self.b = bs
        self.k = k
        self.m1 = m1 
        self.m2 = m2
        pts_dom, pts_bd = split_domain_boundary(train_pts)
        self.pts_dom = pts_dom #.requires_grad_(True)
        self.pts_bd = pts_bd
    
    def feat_extract(self, x):
        return F.relu(x @ self.W.t() + self.b) ** self.k

    def target(self, x):
        z = torch.sin(self.m1 * pi*x[:,0:1])*torch.sin(self.m2 * pi*x[:,1:2])
        return z 
    
    def rhs(self, x):
        u = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        coeff = 1 + (self.m1 * pi) ** 2 + (self.m2 * pi) ** 2
        return coeff * u

    def lapfeat(self, x):
        """
        Analytic Laplacian of ReLU^k features
        x: (N, d)
        Returns: (N, m)
        """
        # pre-activation
        z = x @ self.W.t() + self.b    # (N, m)

        # mask for ReLU active region
        active = (z > 0).float()

        if self.k < 2:
            # Laplacian is identically zero if k=0 or 1
            return torch.zeros_like(z)

        # squared row norms of W: (m,)
        W_norm2 = (self.W**2).sum(dim=1)   # (m,)
        # expand to (N, m)
        W_norm2 = W_norm2.unsqueeze(0).expand_as(z)

        lap = self.k * (self.k - 1) * (z ** (self.k - 2)) * W_norm2 * active
        return lap

    def assemble(self):
        gs = self.feat_extract(self.pts_dom)
        lap_gs = self.lapfeat(self.pts_dom)
        A_dom = -lap_gs + gs
        b_dom = self.rhs(self.pts_dom)
        A_bd = self.feat_extract(self.pts_bd)
        b_bd = self.target(self.pts_bd)
        A = torch.concat([A_dom, A_bd])
        b = torch.concat([b_dom, b_bd])       
        return A, b

    def solve(self):
        A, b = self.assemble()

        c = 100.0 
        for i in range(len(A)):
            max_a = abs(A[i,:]).max()
            max_b = A[i,:].max()
            if max_a != max_b: 
                ratio = -c/max_a
                A[i,:] = A[i,:]*ratio
                b[i] = b[i]*ratio
            else: 
                ratio = c/max_a
                A[i,:] = A[i,:]*ratio
                b[i] = b[i]*ratio
        alpha = scipy.linalg.lstsq(A.numpy(),b.numpy())[0]
        self.alpha = torch.tensor(alpha)
        
    def forward(self, x):
        return self.feat_extract(x) @ self.alpha
    
    def eval(self, x):
        y_ref = self.target(x)
        y_pred = self.forward(x)
        rel_errl2 = (y_pred - y_ref).norm() / y_ref.norm()
        return rel_errl2.item()
    
class PouShallowReLUkSolver:
    def __init__(
        self, ws, bs, pts, k=1, l=1, sigma=0.01, m1=1, m2=1):
        
        self.n = ws.shape[0]
        self.d = ws.shape[1]
        self.k = k
        self.l = l
        self.sigma = sigma
        self.m1 = m1 
        self.m2 = m2
        self.pts = pts
        cntrs, rng = self.pou_init()
        self.cntrs, self.rng = cntrs, rng

        self.J = self.cntrs.shape[0]
        self.W = ws.unsqueeze(0).repeat(self.J, 1, 1)   # [J, n, d]
        self.b = bs.unsqueeze(0).repeat(self.J, 1)      # [J, n]

        pts_dom, pts_bd = split_domain_boundary(pts)
        self.pts_dom = pts_dom #.requires_grad_(True)
        self.pts_bd = pts_bd
    
    def map_unit(self, x):
        '''
        Inputs:
            x : (N, d)
            cntrs: (J, d)
            rng : float
        Returns:
            x_ : (N, J, d)
        '''
        x_ = (x.unsqueeze(0) - self.cntrs.unsqueeze(1) + self.rng)/(2 * self.rng)
        return x_
    
    def pou_init(self):
        rng = 1 / 2**(self.l)
        cntr_1d = torch.linspace(0, 1, 2**(self.l-1)+1).reshape(-1, 1)
        grid = list(itertools.product(cntr_1d, repeat=2)) # 2 for 2D input
        cntrs = torch.tensor(grid)
        return cntrs, rng

    def pou_extract(self, x):
        '''
        Inputs:
            x : (N, D) tensor
            cntrs : (J, D) tensor 
            rng : scalar (float or tensor)
            sigma : scalar (float or tensor)
        Returns:
            (J, N) tensor
        '''
        def phi(z):
            return 1 / (1 + torch.exp(-z))

        # Expand dimensions for broadcasting: (N,1,D) - (1,M,D) → (N,M,D)
        a = self.cntrs.unsqueeze(1) - self.rng
        b = self.cntrs.unsqueeze(1) + self.rng
        x_exp = x.unsqueeze(0)

        # Apply PoU formula over each dimension
        vals = phi((x_exp - a) / self.sigma) * phi((b - x_exp) / self.sigma)

        # Product over D → (N,M)
        return vals.prod(dim=2, keepdim=False)

    def feat_extract(self, x):
        pou_feat = self.pou_extract(x) # [J, N]
        x_ = self.map_unit(x)
        local_feats = []
        for j in range(self.J):
            feat = F.relu(x_[j] @ self.W[j].t() + self.b[j]) ** self.k
            local_feat = feat * pou_feat[j].unsqueeze(-1)
            local_feats.append(local_feat)
        return torch.hstack(local_feats)

    def lapfeat(self, x):
        """
        Analytic Laplacian of PoU-weighted ReLU^k features.
        Returns (N, J*n) matching feat_extract's column order.
        """
        eps = 1e-12
        beta = self.sigma  # PoU smoothness
        N, d = x.shape
        assert d == self.d == 2, "This implementation assumes 2D."

        # ---------- PoU: phi, grad_phi, lap_phi ----------
        # shapes: a,b -> (J,1,d); xj -> (1,N,d); broadcast -> (J,N,d)
        a = self.cntrs.unsqueeze(1) - self.rng
        b = self.cntrs.unsqueeze(1) + self.rng
        xj = x.unsqueeze(0)

        def sig(u): return torch.sigmoid(u)
        sL = sig((xj - a)/beta)           # left wall σ((x-a)/β)
        sR = sig((b - xj)/beta)           # right wall σ((b-x)/β)

        p   = sL * sR                      # per-dim PoU factors, (J,N,d)
        sL1 = (1.0/beta)  * sL * (1.0 - sL)
        sR1 = (-1.0/beta) * sR * (1.0 - sR)
        sL2 = (1.0/(beta**2)) * sL * (1.0 - sL) * (1.0 - 2.0*sL)
        sR2 = (1.0/(beta**2)) * sR * (1.0 - sR) * (1.0 - 2.0*sR)

        dp  = sL1 * sR + sL * sR1                         # ∂p/∂x_d
        d2p = sL2 * sR + 2.0*sL1*sR1 + sL * sR2           # ∂²p/∂x_d²

        phi = p.prod(dim=2)                               # (J,N)
        R = phi.unsqueeze(-1) / (p + eps)                 # (J,N,d) = ∏_{i≠d} p_i
        grad_phi = dp * R                                 # (J,N,d)
        lap_phi  = (d2p * R).sum(dim=2)                   # (J,N)

        # ---------- Local ReLU^k on unit-mapped coords ----------
        # map_unit: (J,N,d)
        x_u = self.map_unit(x)

        # z: (J,N,n) using batch matmul via einsum
        z = torch.einsum('jnd,jkd->jnk', x_u, self.W) + self.b[:, None, :]
        active = (z > 0).to(z.dtype)
        h = torch.relu(z)
        g = h ** self.k                                    # (J,N,n)

        # ∇g wrt original x: chain factor 1/(2*rng)
        if self.k >= 1:
            kfac = self.k * (z ** (self.k - 1)) * active  # (J,N,n)
            grad_g = kfac.unsqueeze(-1) * (self.W[:, None, :, :] / (2.0*self.rng))  # (J,N,n,d)
        else:
            grad_g = torch.zeros(z.shape + (d,), dtype=z.dtype, device=z.device)

        # Δg wrt original x: chain factor 1/(4*rng^2)
        if self.k >= 2:
            Wn2 = (self.W ** 2).sum(dim=2) / (4.0 * (self.rng ** 2))  # (J,n)
            lap_g = self.k * (self.k - 1) * (z ** (self.k - 2)) * active * Wn2[:, None, :]  # (J,N,n)
        else:
            lap_g = torch.zeros_like(z)

        # ---------- Δ(φ·g) = φΔg + 2∇φ·∇g + gΔφ ----------
        cross = 2.0 * (grad_phi.unsqueeze(2) * grad_g).sum(dim=3)          # (J,N,n)
        lap_f = phi.unsqueeze(2) * lap_g + cross + g * lap_phi.unsqueeze(2)  # (J,N,n)

        # Flatten to (N, J*n) in the same order as feat_extract's hstack over j
        return lap_f.permute(1, 0, 2).reshape(N, self.J * self.n)

    def target(self, x):
        z = torch.sin(self.m1 * pi*x[:,0:1])*torch.sin(self.m2 * pi*x[:,1:2])
        return z 

    def rhs(self, x):
        u = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        coeff = 1 + (self.m1 * pi) ** 2 + (self.m2 * pi) ** 2
        return coeff * u

    def assemble(self):
        gs = self.feat_extract(self.pts_dom)
        # lap_gs = self.lapfeat(gs, self.pts_dom)
        lap_gs = self.lapfeat(self.pts_dom)        
        A_dom = -lap_gs + gs
        b_dom = self.rhs(self.pts_dom)
        A_bd = self.feat_extract(self.pts_bd)
        b_bd = self.target(self.pts_bd)
        A = torch.concat([A_dom, A_bd])
        b = torch.concat([b_dom, b_bd])       
        return A, b

    def solve(self):
        A, b = self.assemble()

        c = 100.0 
        for i in range(len(A)):
            max_a = abs(A[i,:]).max()
            max_b = A[i,:].max()
            if max_a != max_b: 
                ratio = -c/max_a
                A[i,:] = A[i,:]*ratio
                b[i] = b[i]*ratio
            else: 
                ratio = c/max_a
                A[i,:] = A[i,:]*ratio
                b[i] = b[i]*ratio
        # alpha = scipy.linalg.lstsq(A.detach().numpy(),b.detach().numpy())[0]
        alpha = scipy.linalg.lstsq(A.numpy(),b.numpy())[0]
        self.alpha = torch.tensor(alpha)
        
    def forward(self, x):
        return self.feat_extract(x) @ self.alpha
    
    def eval(self, x):
        y_ref = self.target(x)
        y_pred = self.forward(x)
        rel_errl2 = (y_pred - y_ref).norm() / y_ref.norm()
        return rel_errl2.item()
    
class ShallowTanhFitter:
    def __init__(self, ws, bs, train_pts, m1=1, m2=1):
        self.n = ws.shape[0]
        self.W = ws
        self.b = bs
        self.m1 = m1
        self.m2 = m2
        self.pts = train_pts

    def feat_extract(self, x):
        # tanh activation
        return torch.tanh(x @ self.W.t() + self.b)

    def target(self, x):
        # same target as before
        z = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        return z

    def assemble(self):
        A = self.feat_extract(self.pts)
        b = self.target(self.pts)
        return A, b

    def solve(self):
        A, b = self.assemble()

        # scaling (same as your ReLU version)
        c = 100.0
        for i in range(len(A)):
            max_a = abs(A[i, :]).max()
            max_b = A[i, :].max()
            if max_a != max_b:
                ratio = -c / max_a
            else:
                ratio = c / max_a
            A[i, :] = A[i, :] * ratio
            b[i] = b[i] * ratio

        alpha = scipy.linalg.lstsq(A.numpy(), b.numpy())[0]
        self.alpha = torch.tensor(alpha)

    def forward(self, x):
        return self.feat_extract(x) @ self.alpha

    def eval(self, x):
        y_ref = self.target(x)
        y_pred = self.forward(x)
        rel_errl2 = (y_pred - y_ref).norm() / y_ref.norm()
        return rel_errl2.item()

class PouShallowTanhFitter:
    def __init__(self, ws, bs, pts, l=1, sigma=0.01, m1=1, m2=1):
        self.n = ws.shape[0]
        self.d = ws.shape[1]
        self.l = l
        self.sigma = sigma
        self.m1 = m1
        self.m2 = m2
        self.pts = pts.to(dtype=torch.float64)

        cntrs, rng = self.pou_init()
        self.cntrs, self.rng = cntrs.to(dtype=torch.float64), torch.tensor(rng, dtype=torch.float64)

        self.J = self.cntrs.shape[0]
        self.W = ws.to(dtype=torch.float64).unsqueeze(0).repeat(self.J, 1, 1)   # [J, n, d]
        self.b = bs.to(dtype=torch.float64).unsqueeze(0).repeat(self.J, 1)      # [J, n]

    def map_unit(self, x):
        x = x.to(dtype=torch.float64)
        return (x.unsqueeze(0) - self.cntrs.unsqueeze(1) + self.rng) / (2 * self.rng)

    def pou_init(self):
        rng = 1 / 2**(self.l)
        cntr_1d = torch.linspace(0, 1, 2**(self.l - 1) + 1).reshape(-1, 1)
        grid = list(itertools.product(cntr_1d, repeat=2))  # 2D input
        cntrs = torch.tensor(grid)
        return cntrs, rng

    def pou_extract(self, x):
        def phi(z):
            return 1 / (1 + torch.exp(-z))

        x = x.to(dtype=torch.float64)
        a = self.cntrs.unsqueeze(1) - self.rng
        b = self.cntrs.unsqueeze(1) + self.rng
        x_exp = x.unsqueeze(0)

        vals = phi((x_exp - a) / self.sigma) * phi((b - x_exp) / self.sigma)
        return vals.prod(dim=2, keepdim=False)  # (J, N)

    def feat_extract(self, x):
        pou_feat = self.pou_extract(x)  # [J, N]
        x_ = self.map_unit(x)
        local_feats = []
        for j in range(self.J):
            feat = torch.tanh(x_[j] @ self.W[j].t() + self.b[j])  # tanh activation
            local_feat = feat * pou_feat[j].unsqueeze(-1)
            local_feats.append(local_feat)
        return torch.hstack(local_feats)

    def target(self, x):
        x = x.to(dtype=torch.float64)
        z = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        return z

    def assemble(self):
        A = self.feat_extract(self.pts)
        b = self.target(self.pts)
        return A, b

    def solve(self):
        A, b = self.assemble()

        c = 100.0
        for i in range(len(A)):
            max_a = abs(A[i, :]).max()
            max_b = A[i, :].max()
            if max_a != max_b:
                ratio = -c / max_a
            else:
                ratio = c / max_a
            A[i, :] = A[i, :] * ratio
            b[i] = b[i] * ratio

        alpha = scipy.linalg.lstsq(A.numpy(), b.numpy())[0]
        self.alpha = torch.tensor(alpha, dtype=torch.float64)

    def forward(self, x):
        return self.feat_extract(x) @ self.alpha

    def eval(self, x):
        y_ref = self.target(x)
        y_pred = self.forward(x)
        rel_errl2 = (y_pred - y_ref).norm() / y_ref.norm()
        return rel_errl2.item()


class ShallowTanhSolver:
    def __init__(self, ws, bs, train_pts, m1=1, m2=1):
        self.n = ws.shape[0]
        self.W = ws.to(dtype=torch.float64)
        self.b = bs.to(dtype=torch.float64)
        self.m1 = m1
        self.m2 = m2
        pts_dom, pts_bd = split_domain_boundary(train_pts.to(dtype=torch.float64))
        self.pts_dom = pts_dom
        self.pts_bd = pts_bd

    def feat_extract(self, x):
        x = x.to(dtype=torch.float64)
        return torch.tanh(x @ self.W.t() + self.b)

    def target(self, x):
        x = x.to(dtype=torch.float64)
        z = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        return z

    def rhs(self, x):
        x = x.to(dtype=torch.float64)
        u = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        coeff = 1 + (self.m1 * pi) ** 2 + (self.m2 * pi) ** 2
        return coeff * u

    def lapfeat(self, x):
        """
        Analytic Laplacian of tanh features.
        x: (N, d)
        Returns: (N, m)
        """
        z = x @ self.W.t() + self.b    # (N, m)
        W_norm2 = (self.W**2).sum(dim=1).unsqueeze(0).expand_as(z)  # (N, m)
        tanh_z = torch.tanh(z)
        lap = -2 * tanh_z * (1 - tanh_z**2) * W_norm2
        return lap

    def assemble(self):
        gs = self.feat_extract(self.pts_dom)
        lap_gs = self.lapfeat(self.pts_dom)
        A_dom = -lap_gs + gs
        b_dom = self.rhs(self.pts_dom)

        A_bd = self.feat_extract(self.pts_bd)
        b_bd = self.target(self.pts_bd)

        A = torch.concat([A_dom, A_bd])
        b = torch.concat([b_dom, b_bd])
        return A, b

    def solve(self):
        A, b = self.assemble()

        c = 100.0
        for i in range(len(A)):
            max_a = abs(A[i, :]).max()
            max_b = A[i, :].max()
            if max_a != max_b:
                ratio = -c / max_a
            else:
                ratio = c / max_a
            A[i, :] = A[i, :] * ratio
            b[i] = b[i] * ratio

        alpha = scipy.linalg.lstsq(A.numpy(), b.numpy())[0]
        self.alpha = torch.tensor(alpha, dtype=torch.float64)

    def forward(self, x):
        return self.feat_extract(x) @ self.alpha

    def eval(self, x):
        y_ref = self.target(x)
        y_pred = self.forward(x)
        rel_errl2 = (y_pred - y_ref).norm() / y_ref.norm()
        return rel_errl2.item()

class PouShallowTanhSolver:
    def __init__(self, ws, bs, pts, l=1, sigma=0.01, m1=1, m2=1):
        self.n = ws.shape[0]
        self.d = ws.shape[1]
        self.l = l
        self.sigma = sigma
        self.m1 = m1
        self.m2 = m2
        self.pts = pts.to(dtype=torch.float64)

        cntrs, rng = self.pou_init()
        self.cntrs, self.rng = cntrs.to(dtype=torch.float64), rng

        self.J = self.cntrs.shape[0]
        self.W = ws.to(dtype=torch.float64).unsqueeze(0).repeat(self.J, 1, 1)   # [J, n, d]
        self.b = bs.to(dtype=torch.float64).unsqueeze(0).repeat(self.J, 1)      # [J, n]

        pts_dom, pts_bd = split_domain_boundary(self.pts)
        self.pts_dom = pts_dom
        self.pts_bd = pts_bd

    def map_unit(self, x):
        x_ = (x.unsqueeze(0) - self.cntrs.unsqueeze(1) + self.rng) / (2 * self.rng)
        return x_

    def pou_init(self):
        rng = 1 / 2**(self.l)
        cntr_1d = torch.linspace(0, 1, 2**(self.l-1) + 1).reshape(-1, 1)
        grid = list(itertools.product(cntr_1d, repeat=2))  # 2D input
        cntrs = torch.tensor(grid)
        return cntrs, rng

    def pou_extract(self, x):
        def phi(z):
            return 1 / (1 + torch.exp(-z))

        a = self.cntrs.unsqueeze(1) - self.rng
        b = self.cntrs.unsqueeze(1) + self.rng
        x_exp = x.unsqueeze(0)

        vals = phi((x_exp - a) / self.sigma) * phi((b - x_exp) / self.sigma)
        return vals.prod(dim=2, keepdim=False)  # (J, N)

    def feat_extract(self, x):
        pou_feat = self.pou_extract(x)  # (J, N)
        x_ = self.map_unit(x)  # (J, N, d)

        local_feats = []
        for j in range(self.J):
            feat = torch.tanh(x_[j] @ self.W[j].t() + self.b[j])  # (N, n)
            local_feat = feat * pou_feat[j].unsqueeze(-1)
            local_feats.append(local_feat)
        return torch.hstack(local_feats)

    def lapfeat(self, x):
        eps = 1e-12
        beta = self.sigma
        N, d = x.shape
        assert d == self.d == 2, "This implementation assumes 2D."

        # ---------- PoU Laplacian ----------
        a = self.cntrs.unsqueeze(1) - self.rng
        b = self.cntrs.unsqueeze(1) + self.rng
        xj = x.unsqueeze(0)

        def sig(u): return torch.sigmoid(u)
        sL = sig((xj - a) / beta)
        sR = sig((b - xj) / beta)

        p   = sL * sR
        sL1 = (1.0/beta)  * sL * (1.0 - sL)
        sR1 = (-1.0/beta) * sR * (1.0 - sR)
        sL2 = (1.0/(beta**2)) * sL * (1.0 - sL) * (1.0 - 2.0*sL)
        sR2 = (1.0/(beta**2)) * sR * (1.0 - sR) * (1.0 - 2.0*sR)

        dp  = sL1 * sR + sL * sR1
        d2p = sL2 * sR + 2.0*sL1*sR1 + sL * sR2

        phi = p.prod(dim=2)                  # (J,N)
        R = phi.unsqueeze(-1) / (p + eps)    # (J,N,d)
        grad_phi = dp * R                    # (J,N,d)
        lap_phi  = (d2p * R).sum(dim=2)      # (J,N)

        # ---------- Local tanh features ----------
        x_u = self.map_unit(x)
        z = torch.einsum('jnd,jkd->jnk', x_u, self.W) + self.b[:, None, :]  # (J,N,n)
        tanh_z = torch.tanh(z)

        # grad g wrt x
        grad_g = (1 - tanh_z**2).unsqueeze(-1) * (self.W[:, None, :, :] / (2.0 * self.rng))  # (J,N,n,d)

        # lap g wrt x
        Wn2 = (self.W**2).sum(dim=2) / (4.0 * (self.rng**2))  # (J,n)
        lap_g = -2 * tanh_z * (1 - tanh_z**2) * Wn2[:, None, :]  # (J,N,n)

        # ---------- Δ(φ·g) = φΔg + 2∇φ·∇g + gΔφ ----------
        cross = 2.0 * (grad_phi.unsqueeze(2) * grad_g).sum(dim=3)          # (J,N,n)
        lap_f = phi.unsqueeze(2) * lap_g + cross + tanh_z * lap_phi.unsqueeze(2)  # (J,N,n)

        return lap_f.permute(1, 0, 2).reshape(N, self.J * self.n)

    def target(self, x):
        z = torch.sin(self.m1 * pi * x[:,0:1]) * torch.sin(self.m2 * pi * x[:,1:2])
        return z

    def rhs(self, x):
        u = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        coeff = 1 + (self.m1 * pi) ** 2 + (self.m2 * pi) ** 2
        return coeff * u

    def assemble(self):
        gs = self.feat_extract(self.pts_dom)
        lap_gs = self.lapfeat(self.pts_dom)
        A_dom = -lap_gs + gs
        b_dom = self.rhs(self.pts_dom)
        A_bd = self.feat_extract(self.pts_bd)
        b_bd = self.target(self.pts_bd)
        A = torch.concat([A_dom, A_bd])
        b = torch.concat([b_dom, b_bd])
        return A, b

    def solve(self):
        A, b = self.assemble()

        c = 100.0
        for i in range(len(A)):
            max_a = abs(A[i, :]).max()
            max_b = A[i, :].max()
            if max_a != max_b:
                ratio = -c / max_a
            else:
                ratio = c / max_a
            A[i, :] = A[i, :] * ratio
            b[i] = b[i] * ratio

        alpha = scipy.linalg.lstsq(A.numpy(), b.numpy())[0]
        self.alpha = torch.tensor(alpha, dtype=torch.float64)

    def forward(self, x):
        return self.feat_extract(x) @ self.alpha

    def eval(self, x):
        y_ref = self.target(x)
        y_pred = self.forward(x)
        rel_errl2 = (y_pred - y_ref).norm() / y_ref.norm()
        return rel_errl2.item()
