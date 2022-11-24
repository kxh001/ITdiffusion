import argparse
import json

from diffusionmodel import DiffusionModel
from .unet import UNetModel, WrapUNetModel, WrapUNet2DModel
from diffusers import UNet2DModel
import logger


NUM_CLASSES = 10 # used in sampling, CIFAR-10: 10, ImageNet: 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        wrapped=False,
        iddpm=True,
        soft=False,
        ddpm_config_path="",
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    wrapped,
    iddpm,
    soft,
    ddpm_config_path,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        wrapped=wrapped,
        iddpm=iddpm,
        soft=soft,
        ddpm_config_path=ddpm_config_path,
    )
    diffusion = create_information_theoretic_diffusion(
        model=model,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    wrapped,
    iddpm,
    soft,
    ddpm_config_path,
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    if iddpm:
        if wrapped:
            logger.log("Use wrapped IDDPM model...")
            return WrapUNetModel(
                soft=soft,
                in_channels=3,
                model_channels=num_channels,
                out_channels=(3 if not learn_sigma else 6),
                num_res_blocks=num_res_blocks,
                attention_resolutions=tuple(attention_ds),
                dropout=dropout,
                channel_mult=channel_mult,
                num_classes=(NUM_CLASSES if class_cond else None),
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_heads_upsample=num_heads_upsample,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        else:
            logger.log("Use original IDDPM model...")
            return UNetModel(
                in_channels=3,
                model_channels=num_channels,
                out_channels=(3 if not learn_sigma else 6),
                num_res_blocks=num_res_blocks,
                attention_resolutions=tuple(attention_ds),
                dropout=dropout,
                channel_mult=channel_mult,
                num_classes=(NUM_CLASSES if class_cond else None),
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_heads_upsample=num_heads_upsample,
                use_scale_shift_norm=use_scale_shift_norm,
            )
    else:
        f = open(ddpm_config_path)
        model_config = json.load(f)
        if wrapped:
            logger.log("Use wrapped DDPM model(Hugging Face)...")
            return WrapUNet2DModel(soft=soft, **model_config)
        else:
            logger.log("Use original DDPM model(Hugging Face)...")
            return UNet2DModel(**model_config)

def create_information_theoretic_diffusion(
    model,
):
    return DiffusionModel(
        model = model,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
