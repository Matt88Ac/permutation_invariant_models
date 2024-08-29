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
