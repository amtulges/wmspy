import lmfit
from copy import deepcopy

def sum_args(*args):
    output = args[0]
    for arg in args[1:]:
        output += arg
    return output

def make_dependent_copy(model: lmfit.Model, new_prefix: str) -> lmfit.Model:
    if model.prefix == new_prefix:
        raise ValueError(f'{new_prefix=} must be different from {model.prefix=} in make_dependent_copy. ')
    model_copy = deepcopy(model)
    model_copy.prefix = new_prefix
    params = model_copy.make_params()
    for name, _ in params.items():
        model_copy.set_param_hint(
            name,
            expr = name.replace(new_prefix, model.prefix),
            )
    return model_copy
