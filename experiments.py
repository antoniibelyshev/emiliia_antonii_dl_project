from omegaconf import DictConfig
import hydra
from configs.utils import check_cfg, get_single_experiment_cfg_list
from datasets import datasets_dict
from models import models_dict
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    check_cfg(cfg)

    single_experiment_cfg_list = get_single_experiment_cfg_list(cfg)
    
    use_gpu = cfg.meta.use_gpu and torch.cuda.is_available()

    device = torch.device("cuda" if use_gpu else "cpu")

    wandb.login(key=cfg.meta["wandb_ssh_key"])

    for i, single_cfg in enumerate(single_experiment_cfg_list):
        dataset = datasets_dict[single_cfg["dataset"].pop("name")](**single_cfg["dataset"])
        train_dataloader = dataset.get_dataloader(device)
        valid_dataloader = dataset.get_dataloader(device, valid=True)

        model = models_dict[single_cfg["model"].pop("name")](train_dataloader, valid_dataloader, dataset.nu, dataset.trunc_dim, **single_cfg["model"]).to(device)

        record_cfg = {**single_cfg["dataset"], **single_cfg["model"], **single_cfg["trainer"]}

        if use_gpu:
            single_cfg["trainer"]["gpus"] = 1


        run = wandb.init(
            config=record_cfg,
            **cfg.meta.wandb_init
        )

        trainer = pl.Trainer(**single_cfg["trainer"], logger=WandbLogger())

        trainer.fit(model)

        run.finish()


if __name__ == "__main__":
    main()