
from .base_optimizer import base_optimizer
from .eigenvector_follow import eigenvector_follow
from .lbfgs import lbfgs

__all__ = [
    "construct_optimizer",
    "register_optimizer"
]

optimizer_registry = {
    'eigenvector_follow':eigenvector_follow,
    'lbfgs':lbfgs
}
def register_optimizer(name, opt_class):
    optimizer_registry[name] = opt_class
def resolve_optimizer(method):
    if isinstance(method, str):
        method = optimizer_registry[method]
    return method
def construct_optimizer(base_opt, **etc):
    if isinstance(base_opt, base_optimizer):
        return base_opt.copy()
    if hasattr(base_opt, 'items'): # cheap check
        etc = dict(base_opt, **etc)
        base_opt = base_opt.pop('method')
    return resolve_optimizer(base_opt)(**etc)
