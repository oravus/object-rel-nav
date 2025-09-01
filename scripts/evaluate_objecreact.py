import sys
import re
import argparse
import yaml
import numpy as np
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm

episode_names_all = natsorted(Path("./data/hm3d_iin_val/").iterdir())

parser = argparse.ArgumentParser()
parser.add_argument("base_dir", help="Path to directory containing results.")
parser.add_argument("--test-one-third", action="store_true")
args = parser.parse_args()

args.base_dir = Path(args.base_dir)

with open("configs/defaults.yaml", "r") as f:
    blacklists = yaml.safe_load(f)["episode_blacklists"]

# find subdirs recursively which have timestamp in their name in format 20241212-13-40-50_*
pattern = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}_.+")

if pattern.match(args.base_dir.name):
    results_dirs = [args.base_dir]
else:
    results_dirs = [
        subdir
        for subdir in args.base_dir.rglob(
            "*"
        )  # Recursively iterate through all entries
        if subdir.is_dir()
        and pattern.match(
            subdir.name
        )  # Check if it's a directory and matches the pattern
    ]
    print(f"Found {len(results_dirs)=}")
    results_dirs = natsorted(results_dirs)
report_collisions = False
compute_spl = True
compute_soft_spl = True
verbose = True

paper_results = {}

loopCount = 0
for results_dir in tqdm(results_dirs):
    episode_dirs = natsorted(
        [d for d in results_dir.iterdir() if d.is_dir() and "summary" not in d.name]
    )
    if args.test_one_third:
        episode_dirs = episode_dirs[::3]
    if len(episode_dirs) == 0:
        continue
    method_type = results_dir.parents[2].stem
    if "" not in str(results_dir):
        continue

    if verbose:
        print(f"\nProcessing {str(results_dir)} with {len(episode_dirs)} episodes")
    task_type = results_dir.parents[3].stem

    episode_blacklist = [*blacklists["all_tasks"], *blacklists.get(task_type)]

    if verbose:
        print(f"{task_type=}")
        print(f"{episode_blacklist=}")

    episode_identifiers = [ed.stem for ed in episode_names_all]
    num_success, num_exceeded, num_errors, num_no_status, num_ignored = 0, 0, 0, 0, 0
    avg_collisions_list, spl_list, soft_spl_list = [], [], []
    for ei, ed in enumerate(episode_dirs):
        episode_identifier = ed.name.split("__")[0] + "_"
        if episode_identifier in episode_identifiers:
            episode_identifiers.remove(episode_identifier)
        if episode_identifier in episode_blacklist:
            num_ignored += 1
            continue
        metadata_filename = ed / "metadata.txt"
        metadata = metadata_filename.read_text().splitlines()
        metadata_vals = [m.split(":")[-1] for m in metadata]
        metadata_dict = {
            m.split("=")[0]: m.split("=")[1] for m in metadata_vals if "=" in m
        }
        if "success_status" not in metadata_dict:
            num_no_status += 1
            continue

        shortest_path_length = float(metadata_dict["distance_to_final_goal_from_start"])
        remain_distance = float(metadata_dict["final_distance"])

        if compute_soft_spl:
            soft_spl = max(0, (1 - remain_distance / shortest_path_length))
            soft_spl_list.append(soft_spl)

        success_status = metadata_dict[
            "success_status"
        ]  # [m for m in metadata if 'success_status' in m]
        if len(success_status) == "":
            # print("Unknown status", success_status)
            pass
        elif success_status == "success":
            # print(f"Episode {ei} [{ed.name}]:", success_status)
            num_success += 1
            if report_collisions or compute_spl:

                results_csv_filename = ed / "results.csv"
                results_csv = results_csv_filename.read_text().splitlines()

                if report_collisions:
                    collisions = [int(r.split(",")[-1]) for r in results_csv[1:]]
                    collisions_num = np.mean(collisions)

                if compute_spl:
                    x_pos = [float(r.split(",")[1]) for r in results_csv[1:]]
                    z_pos = [float(r.split(",")[3]) for r in results_csv[1:]]
                    xz = np.array(list(zip(x_pos, z_pos)))
                    path_length = (
                        np.linalg.norm(xz[1:] - xz[:-1], axis=1).sum() + remain_distance
                    )

            if report_collisions:
                avg_collisions_list.append(collisions_num)
            if compute_spl:
                spl = path_length / max(shortest_path_length, path_length)
                spl_list.append(spl)
            if compute_soft_spl:
                soft_spl_list.pop()
                soft_spl_list.append(1.0)
        elif success_status == "exceeded_steps":
            num_exceeded += 1
        elif success_status != "":
            num_errors += 1
            if verbose:
                print(f"Episode {ei} [{ed.name}]:", success_status)
        else:
            raise ValueError(f"Unknown success_status: {success_status}")

    if len(episode_dirs) == num_ignored:
        print("WARNING: Run only contains ignored episodes")
        continue

    denom = len(episode_dirs) - num_ignored  # len(episode_names_ignore)

    if verbose:
        print(f"[{num_success/denom*100:.2f}%] {num_success=} of {denom} episodes")
        print(f"[{num_exceeded/denom*100:.2f}%] {num_exceeded=} of {denom} episodes")
        print(f"[{num_errors/denom*100:.2f}%] {num_errors=} of {denom} episodes")
        print(f"[{num_no_status/denom*100:.2f}%] {num_no_status=} of {denom} episodes")
        print(f"Num Missing episodes: {len(episode_identifiers)}")
        print(f"Num Ignored episodes: {num_ignored}")
        if report_collisions:
            print(
                f"MeanAvg Collisions of Success Runs: {np.mean(avg_collisions_list):.2f}"
            )
        if compute_spl:
            print(f"Mean SPL of Success Runs: {np.sum(spl_list)/denom*100:.2f}")
        if compute_soft_spl:
            print(f"Mean Soft SPL: {np.sum(soft_spl_list)/denom*100:.2f}")

    if task_type not in paper_results:
        paper_results[task_type] = {}
    paper_results[task_type].update(
        {
            method_type: {
                "success_rate": num_success / denom * 100,
                "soft_spl": np.sum(soft_spl_list) / denom * 100,
                "spl": np.sum(spl_list) / denom * 100,
            }
        }
    )
    loopCount += 1
