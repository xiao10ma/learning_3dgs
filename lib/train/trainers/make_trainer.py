from .trainer import Trainer
import imp

def _wrapper_factory(cfg, network, train_loader=None): # 参数 network 代表 nerf 网络
    module = cfg.loss_module
    path = cfg.loss_path
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network, train_loader)   # 返回 nerf + criterion
    return network_wrapper


def make_trainer(cfg, network, train_loader=None):
    network = _wrapper_factory(cfg, network, train_loader)  # 返回的network包含了nerf网络，以及criterion优化器
    return Trainer(network)
