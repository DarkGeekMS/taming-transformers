import glob
import os
import subprocess as sp
import sys
from pathlib import Path
from typing import Literal, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from PIL import Image
from tqdm import tqdm

from scripts.make_samples import get_parser, load_model

seed_everything(42424242)
device: Literal['cuda', 'cpu'] = 'cuda'
first_stage_factor = 16
trained_on_res = 256


def get_resolution(resolution_str: str) -> (Tuple[int, int], Tuple[int, int]):
    if not resolution_str.count(',') == 1:
        raise ValueError("Give resolution as in 'height,width'")
    res_h, res_w = resolution_str.split(',')
    res_h = max(int(res_h), trained_on_res)
    res_w = max(int(res_w), trained_on_res)
    z_h = int(round(res_h / first_stage_factor))
    z_w = int(round(res_w / first_stage_factor))
    return (z_h, z_w), (z_h * first_stage_factor, z_w * first_stage_factor)


def add_arg_to_parser(parser):
    parser.add_argument(
        "-R",
        "--resolution",
        type=str,
        default='256,256',
        help=
        f"give resolution in multiples of {first_stage_factor}, default is '256,256'",
    )
    parser.add_argument(
        "-C",
        "--conditional",
        type=str,
        default='objects_bbox',
        help=f"objects_bbox or objects_center_points",
    )
    parser.add_argument(
        "-N",
        "--n_samples_per_layout",
        type=int,
        default=4,
        help=f"how many samples to generate per layout",
    )
    parser.add_argument(
        "-P",
        "--videos_path",
        type=str,
        default='videos',
        help=f"path to dataset videos",
    )
    parser.add_argument(
        "--save_viz",
        action='store_true',
        help=f"whether to save reconstructions",
    )
    return parser


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = add_arg_to_parser(parser)

    opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths) - paths[::-1].index("logs") + 1
            except ValueError:
                idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(
            glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs + opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"):
                del config["data"]
    config = OmegaConf.merge(*configs, cli)
    desired_z_shape, desired_resolution = get_resolution(opt.resolution)
    conditional = opt.conditional

    print(ckpt)
    gpu = True
    eval_mode = True
    show_config = False
    if show_config:
        print(OmegaConf.to_container(config))

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"Global step: {global_step}")

    videos_path = Path(opt.videos_path)
    print("Videos path: ", videos_path)

    subsets = ['1K', '2K', '3K', '4K', '5K', '6K', '7K']

    outdir = Path(opt.outdir).joinpath("dataset")
    outdir.mkdir(exist_ok=True, parents=True)
    print("Writing dataset to ", outdir)

    for subset in subsets:
        print(f"Processing subset {subset}")
        subset_indir = videos_path.joinpath(subset)
        video_names = os.listdir(subset_indir)
        for video_name in tqdm(video_names):
            video_path = os.path.join(subset_indir, video_name, 'images_4')
            video_outdir = outdir.joinpath(video_name)
            if os.path.exists(video_outdir):
                print(f"Skipping {video_path}")
                continue
            video_outdir.mkdir(exist_ok=True, parents=True)
            video_outdir_frames = video_outdir.joinpath('frames')
            video_outdir_frames.mkdir(exist_ok=True, parents=True)
            video_outdir_labels = video_outdir.joinpath('labels')
            video_outdir_labels.mkdir(exist_ok=True, parents=True)
            video_outdir_viz = video_outdir.joinpath('visualizations')
            video_outdir_viz.mkdir(exist_ok=True, parents=True)
            frames_list = sorted(glob.glob(video_path + '/*.png'))
            print(len(frames_list))
            for frame_id, frame_path in enumerate(frames_list):
                if frame_id % 20 != 0:  # save every 20th frame
                    continue
                original_image = Image.open(frame_path)
                if not original_image.mode == "RGB":
                    original_image = original_image.convert("RGB")
                original_image = original_image.resize((512, 512),
                                                       Image.LANCZOS)
                original_image.save(
                    video_outdir_frames.joinpath(frame_path.split('/')[-1]))
                original_image = np.array(original_image).astype(np.uint8)
                image = (original_image / 127.5 - 1.0).astype(np.float32)
                image = torch.from_numpy(image).to(device).unsqueeze(
                    0).permute(0, 3, 1, 2)
                with torch.no_grad():
                    quant_z, indices = model.encode_to_z(image)
                    output_indices = indices.view(1, desired_z_shape[0],
                                                  desired_z_shape[1])
                    quant_z = quant_z.squeeze(0).cpu().numpy()
                    indices = indices.squeeze(0).cpu().numpy()
                np.savez(video_outdir_labels.joinpath(
                    frame_path.split('/')[-1][:-4] + '.npz'),
                         quant_z=quant_z,
                         indices=indices)
                if opt.save_viz:
                    z_shape = (1, model.first_stage_model.quantize.e_dim,
                               desired_z_shape[0], desired_z_shape[1])
                    reconstructed_image = model.decode_to_img(
                        output_indices, z_shape) * 0.5 + 0.5
                    reconstructed_image = reconstructed_image.cpu().numpy(
                    ).squeeze(0).transpose(1, 2, 0).clip(0.0, 1.0)
                    reconstructed_image = (reconstructed_image * 255).astype(
                        np.uint8)
                    final_viz = np.concatenate(
                        (original_image, reconstructed_image), axis=1)
                    Image.fromarray(final_viz).save(
                        video_outdir_viz.joinpath(frame_path.split('/')[-1]))
