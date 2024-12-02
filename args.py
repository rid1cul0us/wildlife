import math
import utils
import argparse

from omegaconf import OmegaConf


def _apply_scaling_rules_to_cfg(cfg):
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(
            cfg.train.batch_size_per_gpu * utils.get_world_size() / 1024.0
        )
        print(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    else:
        raise NotImplementedError
    return cfg


def _default_args(args):
    """
    Complete args for vit-based models.
    """
    # dinov2_default_config = OmegaConf.load("projects/dinov2/dinov2/configs/ssl_default_config.yaml")
    # default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config_file)
    base_lr = args.lr
    args.lr *= math.sqrt(args.batch_size * utils.get_world_size() / 1024.0)
    print(f"sqrt scaling learning rate; base: {base_lr}, new: {args.lr}")
    # cfg = OmegaConf.merge(default_cfg, cfg)
    # _apply_scaling_rules_to_cfg(cfg)
    # write_config(cfg, args.output_dir)
    return cfg


def args_parse():
    parser = argparse.ArgumentParser(description="ERM")

    # arch
    parser.add_argument(
        "--arch",
        default="convnext_tiny",
        type=str,
        help="model type (default: convnext_tiny)",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help="config filepath of arguments for model building",
    )

    parser.add_argument("--seed", default=1234, type=int, help="seed")
    parser.add_argument(
        "--epochs", default=80, type=int, help="total epochs to run (default 80)"
    )
    parser.add_argument(
        "--batch-size", default=128, type=int, help="batch size (default 32)"
    )
    parser.add_argument(
        "--eval_batch_size",
        default=96,
        type=int,
        help="batch size during inference (default 96)",
    )
    parser.add_argument(
        "--scale",
        nargs="+",
        type=int,
        default=[224, 224],
        help="size in transforms (default: 224)",
    )

    # method
    parser.add_argument(
        "--method",
        default="ERM",
        choices=["ERM", "DU", "NumReweighting", "MaxNumReweighting", "PU", "ADU", "MT"],
        type=str,
        help="method to use (default ERM), DU(CoLoCo), PU(LoCo)",
    )
    parser.add_argument("--weights_path", type=str, default=None, help="losses weights")
    parser.add_argument("--ptw", type=float, default=1.0, help="context task weight")
    parser.add_argument(
        "--bw", type=bool, default=False, help="use weights for batch loss"
    )

    # augment
    parser.add_argument(
        "--train_transform",
        default="base_augment_transform",
        type=str,
        help="which train transform to use (default: base_augment_transform)",
    )
    parser.add_argument(
        "--augment_kwargs",
        nargs="*",
        action=utils.ParseKwargs,
        default={},
        help="keyword arguments for data augmentaion passed as key1=value1 key2=value2",
    )

    # parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
    #                     help='Color jitter factor (default: 0.4)')
    # parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
    #                     help='Use AutoAugment policy. 'v0' or 'original'. ' + '(default: rand-m9-mstd0.5-inc1)'),
    # parser.add_argument('--smoothing', type=float, default=0.1,
    #                     help='Label smoothing (default: 0.1)')
    # parser.add_argument('--train_interpolation', type=str, default='bicubic',
    #                     help='Training interpolation (random, bilinear, bicubic default: 'bicubic')')

    # evaluation augment
    parser.add_argument(
        "--test_transform",
        default="test_transform",
        type=str,
        help="which test transform to use (default: test_transform)",
    )
    parser.add_argument(
        "--test_augment_kwargs",
        nargs="*",
        action=utils.ParseKwargs,
        default={},
        help="keyword arguments for test augment passed as key1=value1 key2=value2",
    )

    # # * Random Erase params
    # parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
    #                     help='Random erase prob (default: 0.25)')
    # parser.add_argument('--remode', type=str, default='pixel',
    #                     help='Random erase mode (default: 'pixel')')
    # parser.add_argument('--recount', type=int, default=1,
    #                     help='Random erase count (default: 1)')
    # parser.add_argument('--resplit', type=str2bool, default=False,
    #                     help='Do not random erase first (clean) augmentation split')

    # mixup option
    parser.add_argument(
        "--mixup",
        action="store_true",
        help="saving model in each epoch if set this flag",
    )
    parser.add_argument(
        "--alpha",  # augment_kwargs
        default=1.0,
        type=float,
        help="mixup interpolation coefficient (default: 1)",
    )

    # loss
    parser.add_argument(
        "--loss",
        default="CrossEntropyLoss",
        type=str,
        help="loss function, use name in torch.optim, (default: CrossEntropyLoss)",
    )
    parser.add_argument(
        "--loss_kwargs",
        nargs="*",
        action=utils.ParseKwargs,
        default={},
        help="keyword arguments for loss function initialization passed as key1=value1 key2=value2",
    )

    # optimizer
    parser.add_argument(
        "--optimizer",
        default="AadmW",
        type=str,
        help="optimizer, use name in torch.optim, (default: AdamW)",
    )
    parser.add_argument(
        "--lr", default=5e-5, type=float, help="initial lr, (default: 5e-5)"
    )
    parser.add_argument("--weight-decay", default=1e-8, type=float)
    parser.add_argument(
        "--optimizer_kwargs",
        nargs="*",
        action=utils.ParseKwargs,
        default={},
        help="keyword arguments for optimizer initialization passed as key1=value1 key2=value2",
    )

    # scheduler
    parser.add_argument(
        "--scheduler",
        default="CosineAnnealingLR",
        type=str,
        help="lr scheduler, use name in torch.optim.lr_scheduler, (default: CosineAnnealingLR)",
    )
    parser.add_argument(
        "--scheduler_kwargs",
        nargs="*",
        action=utils.ParseKwargs,
        default={},
        help="keyword arguments for scheduler initialization passed as key1=value1 key2=value2",
    )

    # strategy
    parser.add_argument(
        "--resample", action="store_true", help="use weighted sample resample"
    )

    # dataset & model selection
    parser.add_argument(
        "--dataset",
        default="cct20",
        type=str,
        choices=[
            "cct20",
            "iwildcam",
            "iwildcam-folder",
            "iwildcamfeature",
            "terrainc",
            "iwildcam_randaug_feature",
        ],
        help="select dataset the code will use",
    )
    parser.add_argument(
        "--in_features",
        default=1024,
        type=int,
        help="dims for feature dataset, 1024 for dinov2 vit_l",
    )
    parser.add_argument(
        "--root_dir", default="data/iwildcam", type=str, help="path to dataset root dir"
    )
    parser.add_argument("--train_split", type=str, help="train split name")
    parser.add_argument(
        "--eval_split",
        type=str,
        nargs="+",
        default=["val", "id_test", "test"],
        help="val and test split names",
    )
    parser.add_argument(
        "--model_selection_split",
        type=str,
        default="all_val",
        choices=["id_val", "val", "all_val"],
        help="which split use to select model",
    )
    parser.add_argument(
        "--model_selection_metric",
        type=str,
        default="acc",
        help="which metric use to select model",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="+",
        default=["acc", "loss"],
        help="metrics the algorithm uses",
    )

    # experiment setup
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--model_path", default=None, type=str, help="model path")
    parser.add_argument(
        "--run_name",
        default="",
        type=str,
        help="result and model save sub dir, default is lr_batchsize_timestamp",
    )
    parser.add_argument(
        "--fine-tuning",
        type=str,
        choices=["clf", "backbone", "all"],
        help="""this arg ignores arg resume. if arg model_path unset,
                            default IMAGENET weights would be used,
                            OPTION clf freezing backbone layers but classfier,
                            OPTION backbone only freezing classfier,
                            OPTION all freezing no layer.""",
    )
    parser.add_argument(
        "--result-dir", default="results/", help="dir path to save results"
    )
    parser.add_argument(
        "--save_each_epoch",
        action="store_true",
        help="saving model in each epoch if set this flag",
    )

    # distributed
    parser.add_argument("--distributed", action="store_true", help="use multiple gpus")
    parser.add_argument("--dist_url", default=None, type=str, help="use multiple gpus")
    # parser.add_argument('--local-rank', default=-1, type=int) # torchrun read from os.environ

    args = parser.parse_args()
    if args.config_file:
        args.cfg = _default_args(args)
    return args
