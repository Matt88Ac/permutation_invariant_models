from typing import Literal

try:
    from Networks.general_layer import GeneralLayer
    from Networks.canonization_net import CanonizationNetMLP
    from Networks.linear_equivariant_net import EquivariantNet
    from Networks.sample_symmetrization_net import SampleSymmetrizationNetMLP
    from Networks.symmetrization_net import SymmetrizationNetMLP

except ImportError:
    from general_layer import GeneralLayer
    from canonization_net import CanonizationNetMLP
    from linear_equivariant_net import EquivariantNet
    from sample_symmetrization_net import SampleSymmetrizationNetMLP
    from symmetrization_net import SymmetrizationNetMLP

MODEL_TYPES = Literal['canonization_mlp', 'symmetrization_mlp', 'sample_symmetrization_mlp', 'equivariant']


def get_models(kind: MODEL_TYPES):
    if kind.lower() == 'canonization_mlp':
        return CanonizationNetMLP
    elif kind.lower() == 'symmetrization_mlp':
        return SymmetrizationNetMLP
    elif kind.lower() == 'sample_symmetrization_mlp':
        return SampleSymmetrizationNetMLP
    elif kind.lower() == 'equivariant':
        return EquivariantNet

    raise NotImplementedError("model undefined; pick one of " + ', '.join(['canonization_mlp', 'symmetrization_mlp',
                                                                           'sample_symmetrization_mlp', 'equivariant']))
