def seed_everything(seed: int = 42):
    import random
    import numpy as np
    import torch
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

import os
import sys
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import logging
import yaml
import traceback

from libs.logger import default_logger

if "LOG_LEVEL" not in os.environ:
    os.environ["LOG_LEVEL"] = "DEBUG"
default_logger.setup_logging(level=os.environ["LOG_LEVEL"])

from libs.common import utils
from libs.experiments import task_setup


def run(args):
    # set up all the paths
    path_dataset = Path(args.path_dataset)
    path_scenes_root_hm3d = path_dataset / "hm3d_v0.2" / args.split
    sh_map = args.sim["sensor_height_map"]
    if args.task_type == "via_alt_goal":
        map_dir = f"hm3d_generated/stretch_maps/hm3d_iin_{args.split}/maps_via_alt_goal"
        if sh_map != 1.31:
            map_dir += f"-sh_{sh_map}/"
    else:
        if sh_map != 1.31:
            map_dir = (
                f"hm3d_generated/stretch_maps/hm3d_iin_{args.split}/height-sh_{sh_map}"
            )
        else:
            map_dir = f"hm3d_iin_{args.split}"

    path_episode_root = path_dataset / map_dir
    print(f"Root path for episodes: {path_episode_root}")

    # Results tracking
    results_summary = {
        "total_episodes": 0,
        "successful_episodes": 0,
        "failed_episodes": 0,
        "success_rate": 0.0,
        "failure_reasons": {},
    }
    if args.log_wandb:
        task_setup.setup_wandb_logging(args)

    path_results_folder = task_setup.init_results_dir_and_save_cfg(args, default_logger)
    print("\nConfig file saved in the results folder!\n")

    preload_data = task_setup.preload_models(args)

    episodes = task_setup.load_run_list(args, path_episode_root)[
        args.start_idx : args.end_idx : args.step_idx
    ]
    if len(episodes) == 0:
        raise ValueError(
            f"No episodes found at {path_episode_root=}. Please check the dataset path and indices."
        )
    print(f"Total episodes to process: {len(episodes)}")

    for ei, path_episode in tqdm(
        enumerate(episodes),
        total=len(episodes),
        desc=f"Processing Episodes (Total: {len(episodes)})",
    ):
        results_summary["total_episodes"] += 1
        episode_name = path_episode.parts[-1].split("_")[0]
        path_scene_hm3d = sorted(path_scenes_root_hm3d.glob(f"*{episode_name}"))[0]
        scene_name_hm3d = str(sorted(path_scene_hm3d.glob("*basis.glb"))[0])

        episode_runner = None
        success_status = None
        try:
            episode_runner = task_setup.Episode(
                args, path_episode, scene_name_hm3d, path_results_folder, preload_data
            )

            if args.plot:
                ax, plt = episode_runner.init_plotting()

            for step in range(args.max_steps):
                if episode_runner.is_done():
                    break

                logger.info(f"\tAt step {step} (ep {ei}): Getting sensor observations")
                observations = episode_runner.sim.get_sensor_observations()
                display_img, depth, semantic_instance_sim = utils.split_observations(
                    observations
                )

                if args.infer_depth:
                    depth = (
                        preload_data["depth_model"].infer(display_img) * 0.44
                    )  # is a scaling factor

                logger.info(f"\tAt step {step} (ep {ei}): Getting goal")
                episode_runner.get_goal(display_img, depth, semantic_instance_sim)

                if not args.infer_traversable:  # override the FastSAM traversable mask
                    episode_runner.traversable_mask = utils.get_traversibility(
                        torch.from_numpy(semantic_instance_sim),
                        episode_runner.traversable_class_indices,
                    ).numpy()

                logger.info(f"\tAt step {step} (ep {ei}): Getting control signal")
                episode_runner.get_control_signal(step, display_img, depth)

                logger.info(f"\tAt step {step} (ep {ei}): Executing Action")
                episode_runner.execute_action()

                if args.plot:
                    episode_runner.plot(
                        ax, plt, step, display_img, depth, semantic_instance_sim
                    )

                if args.log_robot:
                    episode_runner.log_results(step)

                print(f"...Steps completed: {step + 1}/{args.max_steps}", end="\r")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            e_filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            success_status = (
                f"{type(e).__name__}: {e} in {e_filename} #{exc_tb.tb_lineno}"
            )

            if episode_runner is not None:
                episode_runner.success_status = success_status

            if args.except_exit:
                traceback.print_exc()
                if episode_runner:
                    episode_runner.close(step)
                exit(-1)

        if episode_runner is not None:
            episode_runner.close(step)  # need to close vis first to save video to wandb
            if args.log_wandb:
                task_setup.wandb_log_episode(
                    path_episode.name,
                    episode_runner.results_dict,
                    episode_runner.video_cfg["savepath"],
                )
            success_status = episode_runner.success_status

        # Track success and failure statistics
        if success_status is None or "success" in success_status.lower():
            results_summary["successful_episodes"] += 1
        else:
            results_summary["failed_episodes"] += 1
            if success_status not in results_summary["failure_reasons"]:
                results_summary["failure_reasons"][success_status] = 0
            results_summary["failure_reasons"][success_status] += 1

        print(f"Completed with success status: {success_status}")

    results_summary = utils.create_results_summary(
        args, results_summary, path_results_folder
    )

    return results_summary


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        "-c",
        help="Path to the config file",
        default="configs/defaults.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger("[Goal Control]")  # Logger for this script

    args = parse_args()

    config_file = args.config_file
    if not os.path.exists(config_file):
        logger.warning(
            f"Using default config file, create {config_file} to customise the parameters"
        )
        config_file = "defaults.yaml"

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Config File {config_file} params: {config}")
            # pass the config to the args
            for k, v in config.items():
                setattr(args, k, v)

    # unsupported combinations
    if args.reverse and args.task_type != "original":
        raise ValueError("Reverse is only supported for original task type")

    # setup traversable classes for TANGO
    setattr(
        args,
        "traversable_class_names",
        [
            "floor",
            "flooring",
            "floor mat",
            "floor vent",
            "carpet",
            "mat",
            "rug",
            "doormat",
            "shower floor",
            "pavement",
            "ground",
            "tiles",
        ],
    )

    run(args)
