from omegaconf import DictConfig
import hydra
from my_lib.configs import check_cfg, get_single_experiment_cfg_list
from my_lib.datasets import datasets_dict
from my_lib.models import models_dict
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch


@hydra.main(config_path="configs", config_name="single_example", version_base="1.1")
def main(cfg: DictConfig) -> None:
    check_cfg(cfg)

    single_experiment_cfg_list = get_single_experiment_cfg_list(cfg)
    
    use_gpu = cfg.meta.use_gpu and torch.cuda.is_available()

    device = torch.device("cuda" if use_gpu else "cpu")

    wandb.login(key=cfg.meta["wandb_ssh_key"])

    for i, single_cfg in enumerate(single_experiment_cfg_list):
        dataset = datasets_dict[single_cfg["dataset"].pop("name")](device, **single_cfg["dataset"])

        model = models_dict[single_cfg["model"].pop("name")](dataset, dataset.trunc_dim, **single_cfg["model"]).to(device)

        record_cfg = {**single_cfg["dataset"], **single_cfg["model"], **single_cfg["trainer"]}

        if use_gpu:
            single_cfg["trainer"]["gpus"] = 1

        name = cfg.meta.wandb_init.pop("name") + str(i)
        run = wandb.init(
            config=record_cfg,
            name=name,
            **cfg.meta.wandb_init,
        )

        trainer = pl.Trainer(**single_cfg["trainer"], logger=WandbLogger())

        trainer.fit(model)

        valid_X, _ = dataset.get_tensor_data(torch.device("cpu"))
        valid_pred = model(valid_X)
        wandb.log({"final_loss": dataset.final_loss(valid_pred.detach().cpu().numpy())})

        run.finish()


if __name__ == "__main__":
    main()