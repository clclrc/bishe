def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2se':
        from .ggcnn2se import GGCNN2SE
        return GGCNN2SE
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
