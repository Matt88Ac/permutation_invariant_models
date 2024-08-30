from typing import Literal

try:
    from Networks.general_layer import GeneralLayer
    from Networks.canonization_net import CanonizationNetMLP, CanonizationNetPosEncode
    from Networks.linear_equivariant_net import EquivariantNet
    from Networks.sample_symmetrization_net import SampleSymmetrizationNetMLP, SampleSymmetrizationNetPosEncode
    from Networks.symmetrization_net import SymmetrizationNetMLP, SymmetrizationNetPosEncode

except ImportError:
    from general_layer import GeneralLayer
    from canonization_net import CanonizationNetMLP, CanonizationNetPosEncode
    from linear_equivariant_net import EquivariantNet
    from sample_symmetrization_net import SampleSymmetrizationNetMLP, SampleSymmetrizationNetPosEncode
    from symmetrization_net import SymmetrizationNetMLP, SymmetrizationNetPosEncode

MODEL_TYPES = Literal['canonization_mlp', 'symmetrization_mlp', 'sample_symmetrization_mlp', 'canonization_transformer', 'symmetrization_transformer', 'sample_symmetrization_transformer', 'equivariant']


def get_models(kind: MODEL_TYPES):
    if kind.lower() == 'canonization_mlp':
        return CanonizationNetMLP
    elif kind.lower() == 'symmetrization_mlp':
        return SymmetrizationNetMLP
    elif kind.lower() == 'sample_symmetrization_mlp':
        return SampleSymmetrizationNetMLP
    elif kind.lower() == 'equivariant':
        return EquivariantNet
    elif kind.lower() == 'canonization_transformer':
        return CanonizationNetPosEncode
    elif kind.lower() == 'symmetrization_transformer':
        return SymmetrizationNetPosEncode
    elif kind.lower() == 'sample_symmetrization_transformer':
        return SampleSymmetrizationNetPosEncode

    raise NotImplementedError("model undefined; pick one of " + ', '.join(['canonization_mlp',
                                                                           'canonization_transformer',
                                                                           'symmetrization_mlp',
                                                                           'symmetrization_transformer',
                                                                           'sample_symmetrization_transformer',
                                                                           'sample_symmetrization_mlp', 'equivariant']))
