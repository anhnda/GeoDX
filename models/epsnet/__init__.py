from .dualenc import DualEncoderEpsNetwork
from .hamiltonian_bridge import HamiltonianBridgeNetwork

def get_model(config):
    if config.network == 'dualenc':
        return DualEncoderEpsNetwork(config)
    elif config.network == 'hamiltonian_bridge':
        return HamiltonianBridgeNetwork(config)
    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
