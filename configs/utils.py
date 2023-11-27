from itertools import product
from omegaconf.listconfig import ListConfig
import copy


def check_cfg(cfg):
    contains_list, equal_length = check_lists(cfg)
    if cfg.meta.type == "single" and contains_list:
        raise ValueError("List of hyperparameters provided for a single experiment")
    if not (cfg.meta.type == "single" or contains_list):
        raise ValueError("No list of hyperparameters provided for multiple experiments")
    if cfg.meta.type == "simultanious" and not equal_length:
        raise ValueError("Lists of hyperparameters have various lengths which is incompatible with simultanious iteration")


def check_lists(cfg):
    length = None
    equal_length = True

    for cfg_name in cfg:
        for value in cfg[cfg_name].values():
            if isinstance(value, ListConfig):
                if length is None:
                    length = len(value)
                elif length != len(value):
                    equal_length = False
                    break
    
    contains_list = (length is not None)
    return contains_list, equal_length


def get_single_experiment_cfg_list(cfg):
    cfg_dict = {"dataset": dict(cfg.dataset), "model": dict(cfg.model), "trainer": dict(cfg.trainer)}

    if cfg.meta.type == "single":
        return [cfg_dict]

    list_hyperparameters_keys = []
    list_hyperparameters = []
    for cfg_name in cfg_dict.keys():
        for parameter, value in cfg_dict[cfg_name].items():
            if isinstance(value, ListConfig):
                list_hyperparameters_keys.append((cfg_name, parameter))
                list_hyperparameters.append(value)

    if cfg.meta.type == "simultanious":
        iterator = zip(*list_hyperparameters)
    if cfg.meta.type == "all":
        iterator = product(*list_hyperparameters)
    
    single_experiment_cfg_list = []
    for values in iterator:
        for value, (cfg_name, parameter) in zip(values, list_hyperparameters_keys):
            cfg_dict[cfg_name][parameter] = value
        single_experiment_cfg_list.append(copy.deepcopy(cfg_dict))

    return single_experiment_cfg_list