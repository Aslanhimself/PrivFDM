import argparse, os, sys, datetime, glob

from omegaconf import OmegaConf
from packaging import version
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

from ldm.data.util import VirtualBatchWrapper
from ldm.modules.diffusionmodules.util import extract_into_tensor
from ldm.util import instantiate_from_config
from ldm.privacy.myopacus import MyDPLightningDataModule

# Support existing configs referring to names in this file
from callbacks.cuda import CUDACallback                         # noqa: F401
from callbacks.image_logger import ImageLogger                  # noqa: F401
from callbacks.setup import SetupCallback                       # noqa: F401
from ldm.data.util import DataModuleFromConfig, WrappedDataset  # noqa: F401

import copy
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from ldm.models.diffusion.ddpm import DDPM
from collections import Counter


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


# Federated DataModule-dirichlet split non-IID
class FederatedDataModule(LightningDataModule):
    def __init__(self, base_datamodule, client_id, num_clients, batch_size=None, alpha=0.5):
        super().__init__()
        self.base_dm = base_datamodule
        self.client_id = client_id
        self.num_clients = num_clients
        self.batch_size = batch_size or base_datamodule.batch_size
        self.alpha = alpha  # Dirichlet

    def prepare_data(self):
        self.base_dm.prepare_data()

    def setup(self, stage=None):
        self.base_dm.setup(stage)
        full_train_dataset = self.base_dm.train_dataloader().dataset

        client_indices_list = self.dirichlet_split_noniid(
            dataset=full_train_dataset,
            num_clients=self.num_clients,
            alpha=self.alpha,
            seed=42,
        )
        self.train_dataset = Subset(full_train_dataset, client_indices_list[self.client_id])

        full_val_dataset = self.base_dm.val_dataloader().dataset
        val_len = len(full_val_dataset)
        val_indices = np.arange(val_len)
        np.random.shuffle(val_indices)
        val_client_size = val_len // self.num_clients
        val_start = self.client_id * val_client_size
        val_end = (self.client_id + 1) * val_client_size if self.client_id != self.num_clients - 1 else val_len
        self.val_dataset = Subset(full_val_dataset, val_indices[val_start:val_end])

    def dirichlet_split_noniid(self, dataset, num_clients, alpha=0.5, seed=42):
        np.random.seed(seed)
        labels = np.array(dataset.targets)
        num_classes = len(np.unique(labels))

        class_indices = [np.where(labels == y)[0] for y in range(num_classes)]
        for c in class_indices:
            np.random.shuffle(c)

        total_samples = sum(len(c) for c in class_indices)
        samples_per_client = total_samples // num_clients

        client_indices = [[] for _ in range(num_clients)]
        client_class_counts = [0 for _ in range(num_clients)]

        for class_id, idxs in enumerate(class_indices):
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))

            proportions = np.clip(proportions, 1e-6, 1)
            proportions = proportions / proportions.sum()

            available_space = [samples_per_client - len(client_indices[i]) for i in range(num_clients)]
            class_allocation = (proportions * len(idxs)).astype(int)

            while class_allocation.sum() > len(idxs):
                class_allocation[np.argmax(class_allocation)] -= 1
            while class_allocation.sum() < len(idxs):
                class_allocation[np.argmin(class_allocation)] += 1

            idx_ptr = 0
            for client_id in range(num_clients):
                need = available_space[client_id]
                take = min(need, class_allocation[client_id])
                client_indices[client_id].extend(idxs[idx_ptr: idx_ptr + take])
                idx_ptr += take

        min_size = min(len(idxs) for idxs in client_indices)
        client_indices = [np.random.choice(idxs, samples_per_client, replace=False).tolist()
                          if len(idxs) > samples_per_client
                          else idxs + np.random.choice(idxs, samples_per_client - len(idxs)).tolist()
                          for idxs in client_indices]

        return client_indices

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


# --- 2. Utility functions for federated learning ---


def freeze_non_unet(model):
    for name, param in model.named_parameters():
        if "diffusion_model" not in name:
            param.requires_grad = False

def extract_unet_weights(state_dict, phase):
    return {k: v for k, v in state_dict.items()
            if "diffusion_model" in k or "cond_stage_model" in k}

    # if phase == "2":
    #     return {k: v for k, v in state_dict.items()
    #             if "diffusion_model" in k or "cond_stage_model" in k}
    #     # return {k: v for k, v in state_dict.items()
    #     #         if "diffusion_model" in k }
    # else:
    #     return {k: v for k, v in state_dict.items()
    #             if "diffusion_model" in k}

def set_unet_weights(model, unet_weights):
    state_dict = model.state_dict()
    for k in unet_weights:
        state_dict[k] = unet_weights[k]
    model.load_state_dict(state_dict)

def federated_average(weights_list):
    avg = copy.deepcopy(weights_list[0])
    for k in avg:
        for w in weights_list[1:]:
            avg[k] += w[k]
        avg[k] /= len(weights_list)
    return avg


########################################################################################################################
if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(glob.escape(logdir), "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = trainer_config.get("accelerator", "ddp")
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if "gpus" not in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        logger_cfg = lightning_config.get("logger", OmegaConf.create())
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }

        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.get("modelcheckpoint", OmegaConf.create())
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        callbacks_cfg = lightning_config.get("callbacks", OmegaConf.create())

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint': {
                    "target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10000,
                        'save_weights_only': True
                    }
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

########################################################################################################################
        # Define number of clients
        num_clients = 6

        client_models = []
        clients_data1 = []
        clients_data2 = []
        client_trainers1 = []
        client_trainers2 = []

        # Instantiate Phase I Training data
        data1 = instantiate_from_config(config.data)
        data1.batch_size = config.data.batch_size1
        data1.prepare_data()
        data1.setup()
        # Instantiate Phase II Training data
        data2 = instantiate_from_config(config.data)
        data2.batch_size = config.data.batch_size2
        data2.prepare_data()
        data2.setup()

        # Distribute dataset for each client
        for i in range(num_clients):
            fed_data1 = FederatedDataModule(data1, client_id=i, num_clients=num_clients, batch_size=config.data.batch_size1, alpha=0.5)
            fed_data1.prepare_data()
            fed_data1.setup()

            # check data distribution
            # subset = fed_data1.train_dataloader().dataset
            # indices = subset.indices
            # full_dataset = data1.train_dataloader().dataset
            # labels = torch.tensor(full_dataset.targets)[indices]
            # label_counts = Counter(labels.tolist())
            # print(f"Client {i} label distribution:")
            # for label in sorted(label_counts):
            #     print(f"  Class {label}: {label_counts[label]} samples")

            # If using DP with Poisson sampling, wrap the datasets
            dp_config = config.model.params.get("dp_config")
            if dp_config and dp_config.enabled and dp_config.poisson_sampling:
                # print("Using Poisson sampling")
                fed_data1 = MyDPLightningDataModule(fed_data1)
            clients_data1.append(fed_data1)

            fed_dm2 = FederatedDataModule(data2, client_id=i, num_clients=num_clients, batch_size=config.data.batch_size2, alpha=0.5)
            fed_dm2.prepare_data()
            fed_dm2.setup()
            # If using DP with Poisson sampling, wrap the datasets
            dp_config = config.model.params.get("dp_config")
            if dp_config and dp_config.enabled and dp_config.poisson_sampling:
                # print("Using Poisson sampling")
                fed_dm2 = MyDPLightningDataModule(fed_dm2)
                if dp_config.get("max_batch_size", None):
                    print("Using virtual batch size of", dp_config.max_batch_size)
                    fed_dm2 = VirtualBatchWrapper(fed_dm2, dp_config.max_batch_size)
            clients_data2.append(fed_dm2)

            # Create trainer1 for each client
            client_logdir = os.path.join(logdir, f"client_{i}")
            ckpt_dir1 = os.path.join(client_logdir, "checkpoints_phase1")
            os.makedirs(ckpt_dir1, exist_ok=True) 
           
            ckpt_callback1 = instantiate_from_config({
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": ckpt_dir1,
                    "filename": "{epoch:06}",
                    "verbose": True,
                    "save_last": True,
                    "save_top_k": 0,
                    # "monitor": "val/loss_simple_ema"
                }
            })
            trainer_kwargs["callbacks"] = [
                cb if not isinstance(cb, ModelCheckpoint) else ckpt_callback1
                for cb in trainer_kwargs["callbacks"]
            ]
            trainer_kwargs["default_root_dir"] = client_logdir
            trainer1 = Trainer.from_argparse_args(trainer_opt, max_epochs=1, gpus=1, **trainer_kwargs)
            client_trainers1.append(trainer1)

            # Create trainer2 for each client
            ckpt_dir2 = os.path.join(client_logdir, "checkpoints_phase2")
            os.makedirs(ckpt_dir2, exist_ok=True) 
           
            ckpt_callback2 = instantiate_from_config({
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": ckpt_dir2,
                    "filename": "{epoch:06}",
                    "verbose": True,
                    "save_last": True,
                    "save_top_k": 0,
                    # "monitor": "val/loss_simple_ema"
                }
            })
            trainer_kwargs["callbacks"] = [
                cb if not isinstance(cb, ModelCheckpoint) else ckpt_callback2
                for cb in trainer_kwargs["callbacks"]
            ]
            trainer_kwargs["default_root_dir"] = client_logdir
            trainer2 = Trainer.from_argparse_args(trainer_opt, max_epochs=1, gpus=1, **trainer_kwargs)
            client_trainers2.append(trainer2)

            # load model ckpt for unexpected situation
            # ckpt_path = ""
            # ckpt_path = ckpt_path.format(i=i)
            # state = torch.load(ckpt_path, map_location="cuda:0")
            # sd = state.get("state_dict", state)
            # model.load_state_dict(sd, strict=True)

            model.learning_rate1 = config.model.learning_rate1
            model.learning_rate2 = config.model.learning_rate2
            if config.model.params.first_stage_config.client_ckpt is not None:
                model.instantiate_first_stage(config.model.params.first_stage_config, client_index=i)
            client_models.append(copy.deepcopy(model))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if client_trainers2[0].global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "on_signal.ckpt")
                client_trainers2[0].save_checkpoint(ckpt_path)
        def divein(*args, **kwargs):
            if client_trainers2[0].global_rank == 0:
                import pudb
                pudb.set_trace()
        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                # Set Phase 1 once！
                for cid in range(num_clients):
                    client_models[cid].phase = "1"
                for rnd in range(lightning_config.trainer.epochs1):
                    print(f"\n[Phase I Round {rnd+1}] Starting federated training...")
                    weights_list = []
                    for cid in range(num_clients):
                        print(f"#### Training Phase I for client {cid} ####")

                        client_models[cid].client_id = cid

                        client_trainers1[cid].fit(client_models[cid], clients_data1[cid])
                        weights = extract_unet_weights(client_models[cid].state_dict(), client_models[cid].phase)
                        weights_list.append(weights)

                    new_global_weights = federated_average(weights_list)
                    for model in client_models:
                        set_unet_weights(model, new_global_weights)
                    print(f"[Phase I Round {rnd+1}] complete.")

                # Set Phase 2 once！
                for cid in range(num_clients):
                    client_models[cid].phase = "2"
                for rnd in range(lightning_config.trainer.epochs2):
                    print(f"\n[Phase II Round {rnd + 1}] Starting federated training...")
                    weights_list = []
                    for cid in range(num_clients):
                        print(f"#### Training Phase II for client {cid} ####")

                        client_models[cid].client_id = cid

                        client_trainers2[cid].fit(client_models[cid], clients_data2[cid])
                        weights = extract_unet_weights(client_models[cid].state_dict(), client_models[cid].phase)
                        weights_list.append(weights)

                    new_global_weights = federated_average(weights_list)
                    for model in client_models:
                        set_unet_weights(model, new_global_weights)
                    print(f"[Phase II Round {rnd + 1}] Aggregation complete.")

            except Exception:
                client_trainers2[0].save_checkpoint(os.path.join(ckptdir, "on_exception.ckpt"))
                raise
        if not opt.no_test and not client_trainers2[0].interrupted:
            client_trainers2[0].test(client_models[0], clients_data2[0])

    except Exception:
        if opt.debug and client_trainers2[0].global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and client_trainers2[0].global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if client_trainers2[0].global_rank == 0:

            print(client_trainers2[0].profiler.summary())
