"""Squared-epsilon primal SVR with cancellation-resistant Armijo changes.

Same objective, regularized intercept and gradient certificate as solver v2.
Only the line-search objective difference is evaluated differently.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def objective_change(beta,direction,delta,raw,residual,C,epsilon,step):
    moved_raw=raw+step*delta
    moved_residual=np.sign(moved_raw)*np.maximum(np.abs(moved_raw)-epsilon,0)
    same_active=((raw>epsilon)&(moved_raw>epsilon))|((raw<-epsilon)&(moved_raw<-epsilon))
    difference=np.where(same_active,step*delta,moved_residual-residual)
    return (step*float(beta@direction)+.5*step*step*float(direction@direction)
            +C*float(np.sum(2*residual*difference+difference*difference)))


class CountedPrimalSVR:
    def __init__(self,C,epsilon,max_iter=5000,gradient_tolerance=1e-6):
        self.C=C; self.epsilon=epsilon
        self.max_iter=max_iter; self.gradient_tolerance=gradient_tolerance

    def fit_counts(self,unique_X,inverse,y):
        z=np.column_stack([unique_X,np.ones(len(unique_X))])
        beta=np.zeros(z.shape[1]); self.history_=[]
        for iteration in range(self.max_iter):
            raw=(z@beta)[inverse]-y
            residual=np.sign(raw)*np.maximum(np.abs(raw)-self.epsilon,0)
            value=.5*float(beta@beta)+self.C*float(residual@residual)
            sums=np.bincount(inverse,weights=residual,minlength=len(z))
            gradient=beta+2*self.C*(z.T@sums)
            norm=float(np.linalg.norm(gradient))
            entry=dict(objective=value,gradient_norm=norm)
            self.history_.append(entry)
            if norm<=self.gradient_tolerance:
                break
            active=np.bincount(inverse,weights=np.abs(raw)>self.epsilon,minlength=len(z))
            hessian=2*self.C*(z.T@(active[:,None]*z))
            hessian.flat[::len(beta)+1]+=1
            direction=-cho_solve(cho_factor(hessian,lower=True,check_finite=False),gradient,check_finite=False)
            slope=float(gradient@direction)
            assert slope<0 and np.isfinite(direction).all()
            delta=(z@direction)[inverse]
            length=1.
            for backtrack in range(50):
                change=objective_change(beta,direction,delta,raw,residual,self.C,self.epsilon,length)
                if change<=1e-4*length*slope:
                    next_beta=beta+length*direction
                    if np.array_equal(beta,next_beta):
                        raise RuntimeError('No representable Newton update before gradient certificate')
                    beta=next_beta
                    entry.update(step=length,backtracks=backtrack,objective_change=change)
                    break
                length*=.5
            else:
                raise RuntimeError('Stable Newton line search failed before gradient certificate')
        else:
            raise RuntimeError('Newton iteration limit reached without gradient certificate')
        self.coef_=beta[:-1];self.intercept_=float(beta[-1]);self.n_iter_=iteration
        self.objective_=value;self.gradient_norm_=norm
        self.objective_gap_upper_bound_=norm*norm/2
        return self

    def predict(self,X):
        return X@self.coef_+self.intercept_
