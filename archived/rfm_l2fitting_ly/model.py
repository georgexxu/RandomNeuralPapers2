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
    
    def eval_h1(self, x): 
        if x.requires_grad != True:
            x.requires_grad_(True)
        target_values = self.target(x)
        target_grad = torch.autograd.grad(outputs=target_values, inputs=x, grad_outputs= torch.ones_like(target_values),retain_graph=True, create_graph=True)[0] 
        target_x = target_grad[:,0:1]
        target_y = target_grad[:,1:2] 

        model_values = self.forward(x)
        model_grad = torch.autograd.grad(outputs=model_values, inputs=x, grad_outputs=torch.ones_like(model_values),retain_graph=True, create_graph=True)[0]
        model_x = model_grad[:,0:1]
        model_y = model_grad[:,1:2]
    
        model_grad = torch.autograd.grad(outputs=model_values, inputs=x, grad_outputs=torch.ones_like(model_values))[0] # no need to create or retain_graph, save memory 
        model_x = model_grad[:,0:1]
        model_y = model_grad[:,1:2] 
        return torch.sum((model_x - target_x)**2 +  (model_y - target_y)**2/x.size(0))**0.5 


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
    def eval_h1(self, x): 
        if x.requires_grad != True:
            x.requires_grad_(True)
        target_values = self.target(x)
        target_grad = torch.autograd.grad(outputs=target_values, inputs=x, grad_outputs= torch.ones_like(target_values),retain_graph=True, create_graph=True)[0] 
        target_x = target_grad[:,0:1]
        target_y = target_grad[:,1:2] 

        model_values = self.forward(x)
        model_grad = torch.autograd.grad(outputs=model_values, inputs=x, grad_outputs=torch.ones_like(model_values),retain_graph=True, create_graph=True)[0]
        model_x = model_grad[:,0:1]
        model_y = model_grad[:,1:2]
    
        model_grad = torch.autograd.grad(outputs=model_values, inputs=x, grad_outputs=torch.ones_like(model_values))[0] # no need to create or retain_graph, save memory 
        model_x = model_grad[:,0:1]
        model_y = model_grad[:,1:2] 
        return torch.sum((model_x - target_x)**2 +  (model_y - target_y)**2/x.size(0))**0.5 

import torch
import torch.nn.functional as F
import numpy as np
import scipy.linalg

class PouShallowReLUkSolver2:
    """
    Partition-of-Unity shallow ReLU^k collocation solver for (I - Δ) u = f on [0,1]^2,
    using compact C^2 (not C^3) window functions w_{α,β} built from the quintic S5.
    """

    def __init__(
        self, ws, bs, pts, k=1, l=1, sigma=0.01, m1=1, m2=1, alpha=0.75, beta=1.25
    ):
        """
        Args:
            ws:   (n, d) base weights
            bs:   (n,)   base biases
            pts:  (N, d) collocation points in [0,1]^2 (torch tensor)
            k:    ReLU exponent (ReLU^k)
            l:    PoU level; rng = 2^{-l}
            sigma: kept for compatibility (not used by new PoU)
            m1,m2: frequencies for manufactured solution
            alpha,beta: plateau and support radii in normalized s=(x-c)/rng units
        """
        self.n = ws.shape[0]
        self.d = ws.shape[1]
        self.k = k
        self.l = l
        self.sigma = sigma  # not used by the new PoU; kept for compatibility
        self.m1 = m1
        self.m2 = m2
        self.alpha = float(alpha)
        self.beta = float(beta)

        # store points
        self.pts = pts

        # build PoU grid
        cntrs, rng = self.pou_init()
        self.cntrs, self.rng = cntrs, float(rng)

        self.J = self.cntrs.shape[0]
        self.W = ws.unsqueeze(0).repeat(self.J, 1, 1)   # (J, n, d)
        self.b = bs.unsqueeze(0).repeat(self.J, 1)      # (J, n)

        # split points (user must provide)
        pts_dom, pts_bd = split_domain_boundary(pts)
        self.pts_dom = pts_dom
        self.pts_bd = pts_bd

    # ----------------------- PoU helpers (compact C^2) -----------------------

    @staticmethod
    def S5(t):
        """ Quintic S-curve: C^2, S5(0)=0, S5(1)=1, S5'(0)=S5'(1)=S5''(0)=S5''(1)=0. """
        return 6*t**5 - 15*t**4 + 10*t**3

    @staticmethod
    def S5p(t):
        """ First derivative: 30 t^2 (t-1)^2. """
        return 30*t**2*(t-1)**2

    @staticmethod
    def S5pp(t):
        """ Second derivative: 60 t (t-1) (2t-1). """
        return 60*t*(t-1)*(2*t-1)

    def pou_init(self):
        """
        Uniform 2D grid of patch centers; rng = 2^{-l}.
        (2^{l-1}+1) nodes per axis.
        """
        rng = 1.0 / (2 ** self.l)
        cntr_1d = torch.linspace(0.0, 1.0, 2**(self.l - 1) + 1)
        cntrs = torch.cartesian_prod(cntr_1d, cntr_1d)  # (J, 2)
        return cntrs, rng

    def map_unit(self, x):
        """
        Map x into each patch's local unit box [0,1]^d:
            x_unit = (x - c + rng) / (2*rng)
        Returns: (J, N, d)
        """
        return (x.unsqueeze(0) - self.cntrs.unsqueeze(1) + self.rng) / (2.0 * self.rng)

    def pou_extract(self, x):
        """
        φ_j(x) = ∏_{k=1}^d w_{α,β}( (x_k - c_{j,k}) / rng ), with
            w_{α,β}(s) = 1,                        |s| <= α
                        = 1 - S5((|s|-α)/(β-α)),   α < |s| < β
                        = 0,                        |s| >= β
        Returns: (J, N)
        """
        J, d = self.cntrs.shape
        xj = x.unsqueeze(0).expand(J, -1, -1)         # (J, N, d)
        cj = self.cntrs.unsqueeze(1)                  # (J, 1, d)
        s = (xj - cj) / self.rng                      # normalized coords (J, N, d)
        u = s.abs()

        alpha, beta = self.alpha, self.beta
        flat = (u <= alpha)
        trans = (u > alpha) & (u < beta)

        w = torch.zeros_like(u)
        w[flat] = 1.0

        if trans.any():
            t = torch.zeros_like(u)
            t[trans] = (u[trans] - alpha) / (beta - alpha)
            S5 = self.S5(t)
            w[trans] = 1.0 - S5[trans]

        phi = w.prod(dim=2)                           # (J, N)
        return phi

    # ----------------------- Features & Laplacian -----------------------

    def feat_extract(self, x):
        """
        Localized shallow ReLU^k features:
            For each patch j:
              z_j(x) = x_unit W_j^T + b_j
              g_j(x) = ReLU(z_j)^k
              φ_j(x) from PoU above
            Return concatenation over patches: (N, J*n)
        """
        pou_feat = self.pou_extract(x)                # (J, N)
        x_u = self.map_unit(x)                        # (J, N, d)

        local_feats = []
        for j in range(self.J):
            z = x_u[j] @ self.W[j].t() + self.b[j]    # (N, n)
            feat = F.relu(z) ** self.k                # (N, n)
            local_feat = feat * pou_feat[j].unsqueeze(-1)
            local_feats.append(local_feat)
        return torch.hstack(local_feats)              # (N, J*n)

    def lapfeat(self, x):
        """
        Analytic Laplacian of PoU-weighted ReLU^k features with compact C^2 PoU.
        Δ(φ g) = φ Δg + 2 ∇φ · ∇g + g Δφ
        Returns: (N, J*n)
        """
        eps = 1e-12
        N, d = x.shape
        assert d == self.d == 2, "This implementation assumes 2D."

        J = self.cntrs.shape[0]
        alpha, beta = self.alpha, self.beta

        # ----- PoU pieces per axis: w, w'_x, w''_x -----
        xj = x.unsqueeze(0).expand(J, -1, -1)         # (J, N, d)
        cj = self.cntrs.unsqueeze(1)                  # (J, 1, d)
        s = (xj - cj) / self.rng                      # (J, N, d)
        u = s.abs()
        sign_s = torch.sign(s).clamp(min=-1., max=1.)

        flat = (u <= alpha)
        trans = (u > alpha) & (u < beta)

        w = torch.zeros_like(u)
        wp_x = torch.zeros_like(u)    # dw/dx
        wpp_x = torch.zeros_like(u)   # d2w/dx2

        w[flat] = 1.0
        # flat region derivatives zero already

        if trans.any():
            t = torch.zeros_like(u)
            t[trans] = (u[trans] - alpha) / (beta - alpha)

            S5 = self.S5(t)
            S5p = self.S5p(t)
            S5pp = self.S5pp(t)

            # w = 1 - S5(t)
            w[trans] = 1.0 - S5[trans]
            # dw/dx = -(S5'(t)/(β-α)) * sign(s) / rng
            wp_x[trans] = -(S5p[trans] / (beta - alpha)) * sign_s[trans] / self.rng
            # d2w/dx2 = -(S5''(t)/(β-α)^2) / rng^2
            wpp_x[trans] = -(S5pp[trans] / ((beta - alpha)**2)) / (self.rng**2)

        # φ, ∇φ, Δφ using separable product rules
        phi = w.prod(dim=2)                           # (J, N)
        w_safe = w.clamp_min(eps)
        log_grad = wp_x / w_safe                      # (J, N, d)
        # Δφ = φ * Σ_k [ w''/w + (w'/w)^2 ]
        lap_term = (wpp_x / w_safe + log_grad**2).sum(dim=2)  # (J, N)
        lap_phi = phi * lap_term                                 # (J, N)
        # ∇φ = φ * (w'/w) per axis
        grad_phi = phi.unsqueeze(-1) * log_grad                  # (J, N, d)

        # ----- Local ReLU^k features and their derivatives -----
        x_u = self.map_unit(x)                                   # (J, N, d)
        z = torch.einsum('jnd,jkd->jnk', x_u, self.W) + self.b[:, None, :]  # (J,N,n)
        active = (z > 0).to(z.dtype)

        g = F.relu(z) ** self.k                                  # (J, N, n)

        # ∇g wrt x: chain factor from unit map = 1 / (2*rng)
        if self.k >= 1:
            kfac = self.k * (z ** (self.k - 1)) * active         # (J, N, n)
            grad_g = kfac.unsqueeze(-1) * (self.W[:, None, :, :] / (2.0 * self.rng))  # (J,N,n,d)
        else:
            grad_g = torch.zeros(z.shape + (d,), dtype=z.dtype, device=z.device)

        # Δg wrt x (piecewise-polynomial on active side). For ReLU^k with affine z:
        # Δg = k(k-1) z^{k-2} * 1_{z>0} * ||∇z||^2, with ||∇z||^2 = sum_k (w_k/(2*rng))^2
        if self.k >= 2:
            Wn2 = (self.W ** 2).sum(dim=2) / (4.0 * (self.rng ** 2))      # (J, n)
            lap_g = self.k * (self.k - 1) * (z ** (self.k - 2)) * active * Wn2[:, None, :]  # (J,N,n)
        else:
            lap_g = torch.zeros_like(z)

        # ----- Δ(φ·g) = φΔg + 2∇φ·∇g + gΔφ -----
        cross = 2.0 * (grad_phi.unsqueeze(2) * grad_g).sum(dim=3)         # (J,N,n)
        lap_f = phi.unsqueeze(2) * lap_g + cross + g * lap_phi.unsqueeze(2)  # (J,N,n)

        # Flatten to (N, J*n) in same order as feat_extract
        return lap_f.permute(1, 0, 2).reshape(N, self.J * self.n)

    # ----------------------- PDE pieces & assembly -----------------------

    def target(self, x):
        """ Manufactured solution u(x,y) = sin(m1πx) sin(m2πy). """
        pi = torch.pi
        return torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])

    def rhs(self, x):
        """ f = (I - Δ)u with u as above -> (1 + (m1π)^2 + (m2π)^2) * u """
        pi = torch.pi
        u = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        coeff = 1 + (self.m1 * pi) ** 2 + (self.m2 * pi) ** 2
        return coeff * u

    def assemble(self):
        """
        Overdetermined system:
          - interior: (-ΔG + G) α ≈ f
          - boundary: G α ≈ u
        """
        gs = self.feat_extract(self.pts_dom)           # (N_dom, J*n)
        lap_gs = self.lapfeat(self.pts_dom)            # (N_dom, J*n)
        A_dom = -lap_gs + gs
        b_dom = self.rhs(self.pts_dom)

        A_bd = self.feat_extract(self.pts_bd)          # (N_bd, J*n)
        b_bd = self.target(self.pts_bd)

        A = torch.concat([A_dom, A_bd], dim=0)
        b = torch.concat([b_dom, b_bd], dim=0)
        return A, b

    def solve(self):
        """
        Solve least squares for α with simple per-row scaling heuristic.
        Stores self.alpha as (J*n, 1).
        """
        A, b = self.assemble()
        A = A.clone()
        b = b.clone()

        c = 100.0
        for i in range(A.shape[0]):
            row = A[i, :]
            max_a = row.abs().max()
            if max_a == 0:
                continue
            max_b = row.max()
            ratio = (-c / max_a) if (max_a != max_b) else (c / max_a)
            A[i, :] = row * ratio
            b[i] = b[i] * ratio

        alpha_np = scipy.linalg.lstsq(A.detach().numpy(), b.detach().numpy())[0]
        alpha = torch.tensor(alpha_np, dtype=A.dtype)
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(1)
        self.alpha = alpha

    def forward(self, x):
        return self.feat_extract(x) @ self.alpha

    def eval(self, x):
        y_ref = self.target(x)
        y_pred = self.forward(x)
        rel_errl2 = (y_pred - y_ref).norm() / y_ref.norm()
        return rel_errl2.item()

    def eval_h1(self, x):
        """
        H^1 seminorm error: ||∇u_model - ∇u_target||_2 over given points (empirical).
        """
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        target_values = self.target(x)
        target_grad = torch.autograd.grad(
            outputs=target_values,
            inputs=x,
            grad_outputs=torch.ones_like(target_values),
            retain_graph=True,
            create_graph=True
        )[0]
        target_x = target_grad[:, 0:1]
        target_y = target_grad[:, 1:2]

        model_values = self.forward(x)
        model_grad = torch.autograd.grad(
            outputs=model_values,
            inputs=x,
            grad_outputs=torch.ones_like(model_values),
            retain_graph=False,
            create_graph=False
        )[0]
        model_x = model_grad[:, 0:1]
        model_y = model_grad[:, 1:2]

        return torch.sum((model_x - target_x)**2 + (model_y - target_y)**2) ** 0.5 / x.size(0)


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
        z = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2]) \
            + torch.sin(2 * self.m1 * pi * x[:, 0:1]) * torch.sin(2 * self.m2 * pi * x[:, 1:2]) \
            + torch.sin(4 * self.m1 * pi * x[:, 0:1]) * torch.sin(4 * self.m2 * pi * x[:, 1:2])
        return z

    def rhs(self, x):
        x = x.to(dtype=torch.float64)
        u1 = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        u2 = torch.sin(2 * self.m1 * pi * x[:, 0:1]) * torch.sin(2 * self.m2 * pi * x[:, 1:2])
        u3 = torch.sin(4 * self.m1 * pi * x[:, 0:1]) * torch.sin(4 * self.m2 * pi * x[:, 1:2])
        coeff1 = 1 + (self.m1 * pi) ** 2 + (self.m2 * pi) ** 2
        coeff2 = 1 + (2 * self.m1 * pi) ** 2 + (2 * self.m2 * pi) ** 2
        coeff3 = 1 + (4 * self.m1 * pi) ** 2 + (4 * self.m2 * pi) ** 2 
        return coeff1 * u1 + coeff2 * u2 + coeff3 * u3 

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
        x = x.to(dtype=torch.float64)
        z = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2]) \
            + torch.sin(2 * self.m1 * pi * x[:, 0:1]) * torch.sin(2 * self.m2 * pi * x[:, 1:2]) \
            + torch.sin(4 * self.m1 * pi * x[:, 0:1]) * torch.sin(4 * self.m2 * pi * x[:, 1:2])
        return z

    def rhs(self, x):
        x = x.to(dtype=torch.float64)
        u1 = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        u2 = torch.sin(2 * self.m1 * pi * x[:, 0:1]) * torch.sin(2 * self.m2 * pi * x[:, 1:2])
        u3 = torch.sin(4 * self.m1 * pi * x[:, 0:1]) * torch.sin(4 * self.m2 * pi * x[:, 1:2])
        coeff1 = 1 + (self.m1 * pi) ** 2 + (self.m2 * pi) ** 2
        coeff2 = 1 + (2 * self.m1 * pi) ** 2 + (2 * self.m2 * pi) ** 2
        coeff3 = 1 + (4 * self.m1 * pi) ** 2 + (4 * self.m2 * pi) ** 2 
        return coeff1 * u1 + coeff2 * u2 + coeff3 * u3 

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

## with a new POU function 
class PouShallowTanhSolver2:
    """
    Partition-of-Unity shallow tanh collocation solver for (I - Δ) u = f on [0,1]^2
    using compact C^2 (not C^3) window functions w_{α,β} built from the quintic S5.

    Requirements:
      - torch, numpy, scipy
      - a helper: split_domain_boundary(pts) -> (pts_dom, pts_bd)
    """

    def __init__(self, ws, bs, pts, l=1, sigma=0.01, m1=1, m2=1, alpha=0.75, beta=1.25):
        """
        Args:
            ws:  (n, d) base weights for a shallow network
            bs:  (n,)    base biases
            pts: (N, d)  collocation points in [0,1]^2 (torch tensor)
            l:   PoU level; rng = 2^{-l}
            sigma: kept for compatibility (not used by new PoU)
            m1,m2: integers controlling manufactured solution frequencies
            alpha, beta: plateau and support radii (in normalized s = (x-c)/rng units)
        """
        self.n = ws.shape[0]
        self.d = ws.shape[1]
        self.l = l
        self.sigma = sigma
        self.m1 = m1
        self.m2 = m2
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.pts = pts.to(dtype=torch.float64)

        cntrs, rng = self.pou_init()
        self.cntrs = cntrs.to(dtype=torch.float64)  # (J, d)
        self.rng = float(rng)

        self.J = self.cntrs.shape[0]
        self.W = ws.to(dtype=torch.float64).unsqueeze(0).repeat(self.J, 1, 1)   # [J, n, d]
        self.b = bs.to(dtype=torch.float64).unsqueeze(0).repeat(self.J, 1)      # [J, n]

        pts_dom, pts_bd = split_domain_boundary(self.pts)
        self.pts_dom = pts_dom
        self.pts_bd = pts_bd

    @staticmethod
    def S5(t):
        """ Quintic S-curve: 6 t^5 - 15 t^4 + 10 t^3 (C^2, zero slope/curvature at 0 and 1). """
        return 6*t**5 - 15*t**4 + 10*t**3

    def pou_init(self):
        """
        Build a uniform 2D grid of PoU patch centers and return the half-size rng = 2^{-l}.
        Centers follow the user's original pattern.
        """
        rng = 1.0 / (2 ** self.l)
        # grid has (2^{l-1}+1) nodes per axis
        cntr_1d = torch.linspace(0.0, 1.0, 2**(self.l - 1) + 1)
        # Cartesian product -> (J, 2)
        cntrs = torch.cartesian_prod(cntr_1d, cntr_1d)
        return cntrs, rng

    def map_unit(self, x):
        """
        Map x into each patch's local unit box [0,1]^d:
            x_unit = (x - c + rng) / (2*rng)
        Returns: (J, N, d)
        """
        x = x.to(dtype=torch.float64)
        return (x.unsqueeze(0) - self.cntrs.unsqueeze(1) + self.rng) / (2.0 * self.rng)

    def pou_extract(self, x):
        """
        φ_j(x) = ∏_{k=1}^d w_{α,β}( (x_k - c_{j,k}) / rng ), with
            w_{α,β}(s) = 1,                        |s| <= α
                        = 1 - S5((|s|-α)/(β-α)),   α < |s| < β
                        = 0,                        |s| >= β
        Returns: (J, N)
        """
        x = x.to(dtype=torch.float64)                # (N, d)
        J, d = self.cntrs.shape                      # (J, d)
        rng = self.rng
        alpha = self.alpha
        beta  = self.beta

        xj = x.unsqueeze(0).expand(J, -1, -1)        # (J, N, d)
        cj = self.cntrs.unsqueeze(1)                 # (J, 1, d)
        s  = (xj - cj) / rng                         # normalized coords (J, N, d)
        u  = s.abs()                                 # |s|

        flat  = (u <= alpha)
        trans = (u > alpha) & (u < beta)

        w = torch.zeros_like(u, dtype=torch.float64)
        # flat region
        w[flat] = 1.0
        # transition region
        t = torch.zeros_like(u)
        t[trans] = (u[trans] - alpha) / (beta - alpha)
        S5 = 6*t**5 - 15*t**4 + 10*t**3
        w[trans] = 1.0 - S5[trans]

        # product over axes -> φ_j(x)
        phi = w.prod(dim=2)                          # (J, N)
        return phi

    def feat_extract(self, x):
        """
        Build localized shallow tanh features:
          For each patch j:
            g_j(x) = tanh( x_unit W_j^T + b_j ), then multiply by φ_j(x).
        Returns: (N, J*n)
        """
        x = x.to(dtype=torch.float64)
        pou_feat = self.pou_extract(x)               # (J, N)
        x_unit = self.map_unit(x)                    # (J, N, d)

        local_feats = []
        for j in range(self.J):
            feat = torch.tanh(x_unit[j] @ self.W[j].t() + self.b[j])  # (N, n)
            local_feat = feat * pou_feat[j].unsqueeze(-1)             # (N, n)
            local_feats.append(local_feat)
        return torch.hstack(local_feats)                                 # (N, J*n)

    def lapfeat(self, x):
        """
        Compute Δ(φ·g) with the new compact C^2 PoU using product rule:
            Δ(φ g) = φ Δg + 2 ∇φ·∇g + g Δφ
        Returns: (N, J*n)
        """
        eps = 1e-12
        x = x.to(dtype=torch.float64)
        N, d = x.shape
        assert d == self.d == 2, "This implementation assumes 2D."

        J = self.cntrs.shape[0]
        rng   = self.rng
        alpha = self.alpha
        beta  = self.beta

        # ----- PoU terms: φ, ∇φ, Δφ -----
        xj = x.unsqueeze(0).expand(J, -1, -1)        # (J, N, d)
        cj = self.cntrs.unsqueeze(1)                 # (J, 1, d)
        s  = (xj - cj) / rng                         # (J, N, d)
        u  = s.abs()
        sign_s = torch.sign(s).clamp(min=-1., max=1.)

        flat  = (u <= alpha)
        trans = (u > alpha) & (u < beta)

        w     = torch.zeros_like(u)
        wp_x  = torch.zeros_like(u)  # dw/dx
        wpp_x = torch.zeros_like(u)  # d2w/dx2

        # flat region
        w[flat]     = 1.0
        wp_x[flat]  = 0.0
        wpp_x[flat] = 0.0

        # transition region
        t = torch.zeros_like(u)
        t[trans] = (u[trans] - alpha) / (beta - alpha)

        # S5, S5', S5'': S5'(t)=30 t^2 (t-1)^2, S5''(t)=60 t (t-1) (2t-1)
        S5      = 6*t**5 - 15*t**4 + 10*t**3
        S5p     = 30*t**2*(t-1)**2
        S5pp    = 60*t*(t-1)*(2*t-1)

        w[trans]      = 1.0 - S5[trans]
        # dw/dx = -(S5'(t)/(β-α)) * sign(s) / rng
        wp_x[trans]   = -(S5p[trans] / (beta - alpha)) * sign_s[trans] / rng
        # d2w/dx2 = -(S5''(t)/(β-α)^2) / rng^2
        wpp_x[trans]  = -(S5pp[trans] / ((beta - alpha)**2)) / (rng**2)

        # φ = ∏_k w_k
        phi = w.prod(dim=2)                          # (J, N)

        # Safe ratios for log-grad terms
        w_safe   = w.clamp_min(eps)
        log_grad = (wp_x / w_safe)                   # (J, N, d)

        # Δφ = φ * Σ_k [ w''/w + (w'/w)^2 ]
        lap_term = (wpp_x / w_safe + log_grad**2).sum(dim=2)   # (J, N)
        lap_phi  = phi * lap_term                               # (J, N)

        # ∇φ = φ * (w'/w) (per axis)
        grad_phi = phi.unsqueeze(-1) * log_grad                 # (J, N, d)

        # ----- Local tanh features g and their grads/laplacians -----
        x_u = self.map_unit(x)                                  # (J, N, d)
        z = torch.einsum('jnd,jkd->jnk', x_u, self.W) + self.b[:, None, :]  # (J,N,n)
        tanh_z = torch.tanh(z)

        # ∇g wrt x: chain from unit map => / (2*rng) along each axis
        grad_g = (1 - tanh_z**2).unsqueeze(-1) * (self.W[:, None, :, :] / (2.0 * rng))  # (J,N,n,d)

        # Δg wrt x: ∑_k ∂_kk tanh(z) with z affine in x, so Δg = -2 tanh(z)(1-tanh^2 z) ||∇z||^2
        Wn2 = (self.W**2).sum(dim=2) / (4.0 * (rng**2))         # (J,n)
        lap_g = -2 * tanh_z * (1 - tanh_z**2) * Wn2[:, None, :] # (J,N,n)

        # ----- Δ(φ·g) = φΔg + 2∇φ·∇g + gΔφ -----
        cross = 2.0 * (grad_phi.unsqueeze(2) * grad_g).sum(dim=3)        # (J,N,n)
        lap_f = phi.unsqueeze(2) * lap_g + cross + tanh_z * lap_phi.unsqueeze(2)  # (J,N,n)

        return lap_f.permute(1, 0, 2).reshape(N, self.J * self.n)

    def target(self, x):
        """ Manufactured solution u on [0,1]^2 for evaluation and boundary conditions. """
        x = x.to(dtype=torch.float64)
        pi = torch.pi
        z = (torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
             + torch.sin(2 * self.m1 * pi * x[:, 0:1]) * torch.sin(2 * self.m2 * pi * x[:, 1:2])
             + torch.sin(4 * self.m1 * pi * x[:, 0:1]) * torch.sin(4 * self.m2 * pi * x[:, 1:2]))
        return z

    def rhs(self, x):
        """ f = (I - Δ)u for the manufactured solution. """
        x = x.to(dtype=torch.float64)
        pi = torch.pi
        u1 = torch.sin(self.m1 * pi * x[:, 0:1]) * torch.sin(self.m2 * pi * x[:, 1:2])
        u2 = torch.sin(2 * self.m1 * pi * x[:, 0:1]) * torch.sin(2 * self.m2 * pi * x[:, 1:2])
        u3 = torch.sin(4 * self.m1 * pi * x[:, 0:1]) * torch.sin(4 * self.m2 * pi * x[:, 1:2])
        coeff1 = 1 + (self.m1 * pi) ** 2 + (self.m2 * pi) ** 2
        coeff2 = 1 + (2 * self.m1 * pi) ** 2 + (2 * self.m2 * pi) ** 2
        coeff3 = 1 + (4 * self.m1 * pi) ** 2 + (4 * self.m2 * pi) ** 2
        return coeff1 * u1 + coeff2 * u2 + coeff3 * u3

    def assemble(self):
        """
        Build the overdetermined linear system A α ≈ b:
          - interior rows: (-ΔG + G) α ≈ f
          - boundary rows: G α ≈ u
        Returns:
            A: (N_dom + N_bd, J*n)
            b: (N_dom + N_bd, 1)
        """
        gs = self.feat_extract(self.pts_dom)           # (N_dom, J*n)
        lap_gs = self.lapfeat(self.pts_dom)            # (N_dom, J*n)
        A_dom = -lap_gs + gs
        b_dom = self.rhs(self.pts_dom)

        A_bd = self.feat_extract(self.pts_bd)          # (N_bd, J*n)
        b_bd = self.target(self.pts_bd)

        A = torch.concat([A_dom, A_bd], dim=0)
        b = torch.concat([b_dom, b_bd], dim=0)
        return A, b

    def solve(self):
        """
        Solve least squares for α. Includes a simple per-row scaling heuristic.
        Stores self.alpha as (J*n, 1) torch.float64.
        """
        A, b = self.assemble()
        A = A.clone()
        b = b.clone()

        c = 100.0
        for i in range(A.shape[0]):
            row = A[i, :]
            max_a = row.abs().max()
            if max_a == 0:
                continue
            max_b = row.max()
            ratio = (-c / max_a) if (max_a != max_b) else (c / max_a)
            A[i, :] = row * ratio
            b[i] = b[i] * ratio

        alpha_np = scipy.linalg.lstsq(A.numpy(), b.numpy())[0]  # (J*n, 1) or (J*n,)
        alpha_t = torch.tensor(alpha_np, dtype=torch.float64)
        if alpha_t.ndim == 1:
            alpha_t = alpha_t.unsqueeze(1)
        self.alpha = alpha_t  # (J*n, 1)

    def forward(self, x):
        """ Predict u(x) = G(x) α. """
        return self.feat_extract(x) @ self.alpha

    def eval(self, x):
        """ Relative L2 error on points x against manufactured target. """
        y_ref = self.target(x)
        y_pred = self.forward(x)
        rel_errl2 = (y_pred - y_ref).norm() / y_ref.norm()
        return rel_errl2.item()
