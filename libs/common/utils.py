from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import curses
import datetime
import os
import csv
import quaternion
import numpy as np
import networkx as nx

from PIL import Image
import torch
import matplotlib.pyplot as plt

import habitat_sim
from habitat_sim.utils.common import quat_to_magnum, quat_from_magnum

logger = logging.getLogger(__name__)


def dict_to_args(cfg_dict):
    args = type('', (), {})()
    for k, v in cfg_dict.items():
        setattr(args, k, v)
    return args


def get_default_args():
    args_dict = {
        'method': 'tango',
        'goal_source': 'gt_metric',
        'graph_filename': None,
        'max_start_distance': 'easy',
        'threshold_goal_distance': 0.5,
        'debug': False,
        'reverse': False,
        'max_steps': 500,
        'run_list': '',
        'path_run': '',
        'path_models': None,
        'log_robot': True,
        'save_vis': False,
        'plot': False,
        'infer_depth': False,
        'infer_traversable': False,
        'segmentor': 'fast_sam',
        'task_type': 'original',
        'use_gt_localization': False,
        'cull_qry_instances': False,
        'cull_map_instances': False,
        'cull_map_method': 'sim',
        'env': 'sim',
        'goal_gen': {
            'textLabels': [],

            # matcher
            'matcher_name': 'lightglue',
            'map_matcher_name': 'lightglue',
            'geometric_verification': True,
            'match_area': False,

            # planner
            'goalNodeIdx': None,
            'edge_weight_str': None,
            'use_goal_nbrs': False,
            'plan_da_nbrs': False,
            'preplan_to_goals_only': False,
            'rewrite_graph_with_allPathLengths': False,

            # localizer
            'loc_radius': 4,
            'subsample_ref': 1,
            'reloc_rad_add': 2,
            'reloc_rad_max': 15,
            'min_num_matches': 0,
            'localizedImgIdx': 0,

            # tracker
            'do_track': False,
        },
        'sim': {
            'width': 320,
            'height': 240,
            'hfov': 120,
            'sensor_height': 0.4,
            'sensor_height_map': 1.31

        },
    }
    # args_dict as attributes of args
    return dict_to_args(args_dict)


def get_K_from_parameters(hfov_degree, width, height):
    hfov = np.deg2rad(float(hfov_degree))
    K = np.array([
        [(width / 2.) / np.tan(hfov / 2.), 0., width / 2.],
        [0., (height / 2.) / np.tan(hfov / 2.), height / 2.],
        [0., 0., 1]])
    return K


def get_K_from_agent(agent):
    specs = agent.agent_config.sensor_specifications[0]
    return get_K_from_parameters(specs.hfov, specs.resolution[1], specs.resolution[0])


# Habitat Semantics
def find_annotation_path(scene_path):
    # find split name from among ['train', 'val', 'test', 'minival']
    split = None
    for s in ['train', 'minival', 'val', 'test']:
        if s in scene_path:
            split = s
            path_till_split = scene_path.split(split)[0]
            break
    if split is None:
        return None
    else:
        return f"{path_till_split}/{split}/hm3d_annotated_{split}_basis.scene_dataset_config.json"


def build_intrinsics(image_width: int,
                     image_height: int,
                     field_of_view_radians_u: float,
                     field_of_view_radians_v: Optional[float] = None,
                     device='cpu') -> torch.Tensor:
    if field_of_view_radians_v is None:
        field_of_view_radians_v = field_of_view_radians_u
    center_u = image_width / 2
    center_v = image_height / 2
    fov_u = (image_width / 2.) / np.tan(field_of_view_radians_u / 2.)
    fov_v = (image_height / 2.) / np.tan(field_of_view_radians_v / 2.)
    intrinsics = np.array([
        [fov_u, 0., center_u],
        [0., fov_v, center_v],
        [0., 0., 1]
    ])
    intrinsics = torch.from_numpy(intrinsics).to(device)
    return intrinsics


def split_observations(observations):
    rgb_obs = observations["color_sensor"]
    depth = observations["depth_sensor"]
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    display_img = np.array(rgb_img.convert('RGB'))
    semantic_instance = observations["semantic_sensor"]  # an array of instance ids
    return display_img, depth, semantic_instance.astype(int)


def robohop_to_pixnav_goal_mask(goal_mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    max_depth_indices = np.where(depth == depth[goal_mask == goal_mask.min()].max())
    indices_goal_mask = (goal_mask == goal_mask.min())[max_depth_indices]
    goal_target = np.array(max_depth_indices).T[indices_goal_mask]
    target_x = goal_target[0, 1]
    target_z = goal_target[0, 0]
    min_z = max(target_z - 5, 0)
    max_z = min(target_z + 5, goal_mask.shape[0])
    min_x = max(target_x - 5, 0)
    max_x = min(target_x + 5, goal_mask.shape[1])
    pixnav_goal_mask = np.zeros_like(goal_mask)
    pixnav_goal_mask[min_z:max_z, min_x:max_x] = 255
    return pixnav_goal_mask


def unproject_points(depth: torch.Tensor, intrinsics_inv, homogeneous_pts) -> torch.Tensor:
    unprojected_points = (torch.matmul(intrinsics_inv, homogeneous_pts)).T
    unprojected_points *= depth
    return unprojected_points


def has_collided(sim, previous_agent_state, current_agent_state):
    # Check if a collision occured
    previous_rigid_state = habitat_sim.RigidState(
        quat_to_magnum(previous_agent_state.rotation), previous_agent_state.position
    )
    current_rigid_state = habitat_sim.RigidState(
        quat_to_magnum(current_agent_state.rotation), current_agent_state.position
    )
    dist_moved_before_filter = (
            current_rigid_state.translation - previous_rigid_state.translation
    ).dot()
    end_pos = sim.step_filter(
        previous_rigid_state.translation, current_rigid_state.translation
    )
    dist_moved_after_filter = (
            end_pos - previous_rigid_state.translation
    ).dot()

    # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
    # collision _didn't_ happen. One such case is going up stairs.  Instead,
    # we check to see if the the amount moved after the application of the filter
    # is _less_ than the amount moved before the application of the filter
    EPS = 1e-5
    collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
    return collided


def get_traversibility(semantic: torch.Tensor, traversable_classes: list) -> torch.Tensor:
    return torch.isin(semantic, torch.tensor(traversable_classes)).to(int)


def apply_velocity(vel_control, agent, sim, velocity, steer, time_step):
    # Update position
    forward_vec = habitat_sim.utils.quat_rotate_vector(agent.state.rotation, np.array([0, 0, -1.0]))
    new_position = agent.state.position + forward_vec * velocity

    # Update rotation
    new_rotation = habitat_sim.utils.quat_from_angle_axis(steer, np.array([0, 1.0, 0]))
    new_rotation = new_rotation * agent.state.rotation

    # Step the physics simulation
    # Integrate the velocity and apply the transform.
    # Note: this can be done at a higher frequency for more accuracy
    agent_state = agent.state
    previous_rigid_state = habitat_sim.RigidState(
        quat_to_magnum(agent_state.rotation), agent_state.position
    )

    target_rigid_state = habitat_sim.RigidState(
        quat_to_magnum(new_rotation), new_position
    )

    # manually integrate the rigid state
    target_rigid_state = vel_control.integrate_transform(
        time_step, target_rigid_state
    )

    # snap rigid state to navmesh and set state to object/agent
    # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
    end_pos = sim.step_filter(
        previous_rigid_state.translation, target_rigid_state.translation
    )

    # set the computed state
    agent_state.position = end_pos
    agent_state.rotation = quat_from_magnum(
        target_rigid_state.rotation
    )
    agent.set_state(agent_state)

    # Check if a collision occurred
    dist_moved_before_filter = (
            target_rigid_state.translation - previous_rigid_state.translation
    ).dot()
    dist_moved_after_filter = (
            end_pos - previous_rigid_state.translation
    ).dot()

    # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
    # collision _didn't_ happen. One such case is going up stairs.  Instead,
    # we check to see if the the amount moved after the application of the filter
    # is _less_ than the amount moved before the application of the filter
    EPS = 1e-5
    collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
    # run any dynamics simulation
    sim.step_physics(dt=time_step)

    return agent, sim, collided


def log_control(xi: float, yi: float, thetai: float,
                xj: float, yj: float, thetaj: float,
                distance_error: float, theta_error: float,
                theta_control: float, thetaj_current: float) -> None:
    s = (f'distance error: {distance_error:.11f}, '
         f'tangent error: {(theta_error * 180 / np.pi):.11f}, '
         f'xi: {xi:.11f}, yi: {yi:.11f}, thetai: {(thetai * 180 / np.pi):.11f}, '
         f'xj: {xj:.11f}, yj: {yj:.11f}, thetaj: {(thetaj * 180 / np.pi):.11f}, '
         f'theta control: {(theta_control * 180 / np.pi):.11f}, '
         f'theta cumulative: {(thetaj_current * 180 / np.pi):.11f}')
    logger.info("%s", s)


def initialize_results(
        filename_metadata_episode,
        filename_results_episode,
        args,
        pid_steer_values,
        hfov_radians,
        time_delta,
        velocity_control,
        goal_position,
        traversable_categories
):
    # write metadata
    with open(str(filename_metadata_episode), 'w') as f:
        f.writelines(f'method={args.method}\n'
                     f'inferring_depth={args.infer_depth}\n'
                     f'goal_source={args.goal_source}\n'
                     f'max steps={args.max_steps}\n'
                     f'goal distance threshold={args.threshold_goal_distance}\n'
                     f'steer pid values={pid_steer_values}\n'
                     f'camera fov={(hfov_radians * 180 / np.pi):.2f}\n'
                     f'time_delta={time_delta}\n'
                     f'velocity_control={velocity_control}\n'
                     f'goal position={list(goal_position) if goal_position is not None else ""}\n'
                     f'traversable categories={traversable_categories if traversable_categories is not None else ""}\n')

    with open(str(filename_results_episode), 'a') as f:
        f.writelines(f'step,x,y,z,yaw,distance_to_goal,velocity_control,theta_control,discrete_action,collided\n')
    return


def write_results(filename_results_episode,
                  step,
                  current_robot_state,
                  distance_to_goal,
                  velocity_control,
                  theta_control,
                  collided,
                  discrete_action
                  ) -> None:
    with open(str(filename_results_episode), 'a') as f:
        f.writelines(f'{step},'
                     f'{current_robot_state.position[0] if current_robot_state is not None else ""},'
                     f'{current_robot_state.position[1] if current_robot_state is not None else ""},'
                     f'{current_robot_state.position[2] if current_robot_state is not None else ""},'
                     f'{np.arccos(quaternion.as_rotation_matrix(current_robot_state.rotation)[0, 0]) * 180 / np.pi if current_robot_state is not None else ""},'
                     f'{distance_to_goal},'
                     f'{velocity_control},'
                     f'{theta_control * 180 / np.pi},'
                     f'{discrete_action},'
                     f'{int(collided) if collided is not None else ""}\n')


def write_final_meta_results(
        filename_metadata_episode: Path,
        success_status: str,
        final_distance: float,
        step: int,
        distance_to_final_goal):
    with open(str(filename_metadata_episode), 'a') as f:
        f.writelines(f'success_status={success_status}\n'
                     f'final_distance={final_distance}\n'
                     f'step={step}\n'
                     f'distance_to_final_goal_from_start={distance_to_final_goal}')


def count_edges_with_given_weight(G, edge_weight_str):
    if edge_weight_str is None:
        return len(G.edges())
    return sum([1 for e in G.edges(data=True) if e[2].get(edge_weight_str) is not None])


def get_edge_weight_types(G):
    edge_weight_types = set()
    for e in G.edges(data=True):
        for k in e[2].keys():
            edge_weight_types.add(k)
    return edge_weight_types


def change_edge_attr(G):
    for e in G.edges(data=True):
        if 'margin' in e[2]:
            e[2]['margin'] = 0.0
    return G


def norm_minmax(costs, max_val=1):
    costs = costs - costs.min()
    if costs.max() != 0:
        costs = costs / costs.max()
    return (costs * max_val)


def normalize_pls_new(pls, scale_factor=100, outlier_value=99, new_max_val=None):

    outliers = pls >= outlier_value
    # if all are outliers, set them to zero
    if sum(outliers) == len(pls):
        return np.zeros_like(pls)

    min_val = pls.min()
    if new_max_val is None:
        new_max_val = pls[~outliers].max() + 1
    else:
        assert new_max_val > pls[~outliers].max(), f"{new_max_val} <= {pls[~outliers].max()}"

    # else set outliers to max value of inliers + 1
    # so that when normalized, they are set to 0
    if sum(outliers) > 0:
        pls[outliers] = new_max_val

    # include a dummy value to ensure that new_max_val -> 0 after norm 'even for inliers'
    pls = np.concatenate([pls, [new_max_val]])

    # normalize so that outliers are set to 0; inliers \in (0, scale_factor]
    pls = scale_factor * (new_max_val - pls) / (new_max_val - min_val)
    return pls[:-1]


def normalize_pls(pls, scale_factor=100, outlier_value=99):
    # remove outlier values if exist
    if pls.max() >= outlier_value:
        # if all are outliers, set them to zero
        if pls.min() >= outlier_value:
            pls = np.zeros_like(pls)
            return pls
        # else set outliers to max value of inliers + 1
        # so that when normalized, they are set to 0
        else:
            pls[pls >= outlier_value] = pls[pls < outlier_value].max() + 1
            # include a dummy value to ensure that the size is the same as the below case
            pls = np.concatenate([pls, [pls.max()]])
    # no outliers
    else:
        # include a dummy value to ensure that the max value is same as that with outliers
        pls = np.concatenate([pls, [pls.max() + 1]])

    # normalize so that outliers are set to 0
    # inliers are ranged (0, scale_factor]
    pls = scale_factor * (pls.max() - pls) / (pls.max() - pls.min())
    return pls[:-1]


def modify_graph(G,nodes,edges):
    G2 = nx.Graph()
    G2.add_nodes_from(nodes)
    G2.add_edges_from(edges)
    G2.graph = G.graph.copy()
    print("Number of nodes & edges in G: ", len(G.nodes), len(G.edges))
    print("Number of nodes & edges in G2: ", len(G2.nodes), len(G2.edges))
    print(f"is_connected(G): {nx.is_connected(G)}")
    print(f"is_connected(G2): {nx.is_connected(G2)}")
    return G2

def intersect_tuples(a, b):
    # Convert lists of tuples to structured arrays
    a_arr = np.array(a, dtype=[('f1', 'int64'), ('f2', 'int64')])
    b_arr = np.array(b, dtype=[('f1', 'int64'), ('f2', 'int64')])

    # Find the intersection
    intersection = np.intersect1d(a_arr, b_arr)

    # Convert the structured arrays back to list of tuples
    return [tuple(row) for row in intersection]

def getSplitEdgeLists(G,flipSim=True):
    if not flipSim:
        raise NotImplementedError
    intraImage_edges = [e for e in G.edges(data=True) if 'sim' not in e[2]]
    da_edges = [(e[0],e[1],{'sim':1-e[2]['sim']}) for e in G.edges(data=True) if 'sim' in e[2]]
    temporal_edges = [(e[0],e[1],{'sim':1-e[2]['sim']}) for e in G.graph['temporalEdges']]

    # find intersection between da_edges and temporal_edges
    da_edges_noAttr = [tuple(sorted((e[0],e[1]))) for e in da_edges]
    temporal_edges_noAttr = [tuple(sorted((e[0],e[1]))) for e in temporal_edges]
    intersection = intersect_tuples(da_edges_noAttr,temporal_edges_noAttr)
    numCommon = len(intersection)

    print(f"Number of intraImage_edges: {len(intraImage_edges)}")
    print(f"Number of da_edges: {len(da_edges)}")
    print(f"Number of temporal_edges: {len(temporal_edges)}")
    print(f"Number of non-intersecting edges (ideally 0): {len(temporal_edges)-numCommon}")

    return intraImage_edges, da_edges, temporal_edges

def mask_to_rle_numpy(array: np.ndarray) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = array.shape
    array = np.transpose(array, (0, 2, 1)).reshape(b, -1)

    # Compute change indices
    diff = array[:, 1:] != array[:, :-1]
    change_indices = np.nonzero(diff)

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[1][change_indices[0] == i]
        cur_idxs = np.concatenate(
            [
                np.array([0], dtype=cur_idxs.dtype),
                cur_idxs + 1,
                np.array([h * w], dtype=cur_idxs.dtype),
            ]
        )
        btw_idxs = np.diff(cur_idxs)
        counts = [] if array[i, 0] == 0 else [0]
        counts.extend(btw_idxs.tolist())
        out.append({"size": [h, w], "counts": counts})
    return out

def rle_to_mask(rle) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order

def nodes2key(nodeInds, key, G=None):
    _key = key
    if key == 'coords':
        _key = 'segmentation'
    if isinstance(nodeInds[0],dict):
        if _key == 'segmentation' and type(nodeInds[0][_key]) == dict:
            values = np.array([rle_to_mask(n[_key]) for n in nodeInds])
        else:
            values = np.array([n[_key] for n in nodeInds])
    else:
        assert G is not None, "nodes can either be dict or indices of nx.Graph"
        if _key == 'segmentation' and type(G.nodes[nodeInds[0]][_key]) == dict:
            values = np.array([rle_to_mask(G.nodes[n][_key]) for n in nodeInds])
        else:
            values = np.array([G.nodes[n][_key] for n in nodeInds])
    if key == 'coords':
        values = np.array([np.array(np.nonzero(v)).mean(1)[::-1].astype(int) for v in values])
    return values


def get_sim_settings(scene, default_agent=0, sensor_height=1.5, width=256, height=256, hfov=90):
    sim_settings = {
        "scene": scene,  # Scene path
        "default_agent": default_agent,  # Index of the default agent
        "sensor_height": sensor_height,  # Height of sensors in meters, relative to the agent
        "width": width,  # Spatial resolution of the observations
        "height": height,
        "hfov": hfov
    }
    return sim_settings


# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    if "scene_dataset_config_file" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    else:
        annotConfigPath = findAnnotationPath(settings["scene"])
        if annotConfigPath is not None:
            print(f"Annotation file found: {annotConfigPath}")
            sim_cfg.scene_dataset_config_file = annotConfigPath
        else:
            print(f"Annotation file not found for {settings['scene']}")

    # agent
    hardware_config = habitat_sim.agent.AgentConfiguration()
    # # Modify the attributes you need
    # hardware_config.height = 20  # Setting the height to 1.6 meters
    # hardware_config.radius = 10  # Setting the radius to 0.2 meters
    # discrete actions defined for objectnav task in habitat-lab/habitat/config/habitat/task/objectnav.yaml
    custom_action_dict = {'stop': habitat_sim.ActionSpec(name='move_forward', actuation=habitat_sim.ActuationSpec(amount=0))}
    for k in hardware_config.action_space.keys():
        custom_action_dict[k] = hardware_config.action_space[k]
    custom_action_dict['look_up'] = habitat_sim.ActionSpec(name='look_up',
                                                           actuation=habitat_sim.ActuationSpec(amount=30))
    custom_action_dict['look_down'] = habitat_sim.ActionSpec(name='look_down',
                                                             actuation=habitat_sim.ActuationSpec(amount=30))

    hardware_config.action_space = custom_action_dict
    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    # add depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.hfov = settings["hfov"]

    hardware_config.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [hardware_config])


def get_sim_agent(test_scene, updateNavMesh=False, agent_radius=0.75, width=320, height=240, hfov=90, sensor_height=1.5):
    sim_settings = get_sim_settings(scene=test_scene, width=width, height=height,  hfov=hfov, sensor_height=sensor_height)
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    # agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
    sim.pathfinder.seed(42)
    agent_state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(agent_state)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

    if updateNavMesh:
        # update navmesh to avoid tight spaces
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = agent_radius
        navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        # sim_topdown_map = sim.pathfinder.get_topdown_view(0.1, 0)

    return sim, agent, action_names


def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    # print(np.array(rgb_img))

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=True)


def navigateAndSee(action, action_names, sim, display=False):
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)
        if display:
            display_sample(observations["color_sensor"])


# Function to translate keyboard commands to action strings
def map_keyB2Act(key_command):
    if key_command == 'w':
        action = 'move_forward'
    elif key_command == 'a':
        action = 'turn_left'
    elif key_command == 'd':
        action = 'turn_right'
    else:
        return None
    return action


def get_kb_command():
    stdscr = curses.initscr()
    curses.cbreak()
    stdscr.keypad(1)

    key_command = stdscr.getch()
    key_mapping = {
        ord('w'): 'w',
        ord('a'): 'a',
        ord('d'): 'd',
        curses.KEY_UP: 'w',
        curses.KEY_LEFT: 'a',
        curses.KEY_RIGHT: 'd'
    }
    command = key_mapping.get(key_command)

    curses.nocbreak()
    stdscr.keypad(0)
    curses.echo()
    curses.endwin()

    return command


def createTimestampedFolderPath(outdir, prefix, subfolder="", excTime=False):
    """
    Create a folder with a timestamped name in the outdir
    :param outdir: where to create the folder
    :param prefix: prefix for the folder name
    :param subfolder: subfolder name, can be a list of subfolders
    :return: paths to the created folder and subfolders
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S%f')
    if excTime: formatted_time = ""
    folder_path = f'{outdir}/{prefix}_{formatted_time}'
    if type(subfolder) == str:
        subfolder = [subfolder]
    sfPaths = []
    for sf in subfolder:
        subfolder_path = f'{outdir}/{prefix}_{formatted_time}/{sf}'
        os.makedirs(subfolder_path, exist_ok=True)
        sfPaths.append(subfolder_path)
    return folder_path, *sfPaths


def get_autoagent_action(autoagent, currImg, agent_params, time_step):
    autoagent.maintain_history(currImg)
    dists, wayps = [], []
    for mapimg in autoagent.topomap:
        dist, wayp = autoagent.predict_currHistAndGoal(autoagent.currImgHistory, mapimg)
        dists.append(dist)
        wayps.append(wayp)
    ptr = np.argmin(dists)
    # autoagent.updateLocalMap(ptr)
    print(ptr, autoagent.localmapIdx)
    wayp = wayps[min(ptr + 2, len(autoagent.topomap) - 1)][0][2]
    dx, dy = wayp[:2]
    theta = np.arctan(dy / dx) / 3.14 * 180
    v, w = autoagent.waypoint_to_velocity(wayp, agent_params, time_step)
    return v, w, dx, theta


def compute_pose_err(s1, s2):
    """
    Compute the position and rotation error between two agent states
    :param s1: habitat_sim.AgentState
    :param s2: habitat_sim.AgentState
    :return: (float, float) position error, rotation error (degrees)
    """
    pos_err = np.linalg.norm(s1.position - s2.position)
    rot_err = np.rad2deg(quaternion.rotation_intrinsic_distance(s1.rotation, s2.rotation))
    return pos_err, rot_err


# Habitat Semantics
def findAnnotationPath(scenePath):
    # find split name from among ['train', 'val', 'test', 'minival']
    split = None
    for s in ['train', 'minival', 'val', 'test']:  # TODO: 'val' inside 'minival'
        if s in scenePath:
            split = s
            pathTillSplit = scenePath.split(split)[0]
            break
    if split is None:
        return None
    else:
        return f"{pathTillSplit}/{split}/hm3d_annotated_{split}_basis.scene_dataset_config.json"

def print_regions(regions, max_regions=10, max_objects=10):
    region_count = 0
    for region in regions:
        category = region.category.name() if region.category is not None else None
        print(
            f"\t Region id:{region.id}, {category=},"
            f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
        )
        object_count = 0
        for obj in region.objects:
            print(
                f"\t \t Object id:{obj.id}, category:{obj.category.name()},"
                f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            )
            object_count += 1
            if object_count >= max_objects:
                break
        region_count += 1
        if region_count >= max_regions:
            break

def print_scene_recur(scene, limit_output=10):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        print_regions(level.regions, limit_output, limit_output)
    
    if len(scene.levels) == 0:
        print_regions(scene.regions, limit_output, limit_output)

    # # Print semantic annotation information (id, category, bounding box details)
    # # about levels, regions and objects in a hierarchical fashion
    # scene = sim.semantic_scene
    # print_scene_recur(scene)

def obj_id_to_int(obj):
    return int(obj.id.split("_")[-1])

def get_instance_to_category_mapping(semanticScene):
    instance_id_to_label_id = np.array(
        [[obj_id_to_int(obj), obj.category.index()] for obj in semanticScene.objects])
    return instance_id_to_label_id


def get_instance_index_to_name_mapping(semanticScene):
    instance_index_to_name = np.array([[i, obj.category.name()] for i, obj in enumerate(semanticScene.objects)])
    return instance_index_to_name

def get_instance_id_to_region_id_mapping(semantic_scene):
    instance_index_to_region_id = np.array([[obj_id_to_int(obj), int(obj.region.id[1:])] for obj in semantic_scene.objects])

    # check if object ids iterate exactly over total objects
    assert(instance_index_to_region_id[-1, 0] == len(semantic_scene.objects) - 1)
    return instance_index_to_region_id

def get_region_id_to_instance_id_dict(semantic_scene):
    region_id_to_instance_id = {}
    for region in semantic_scene.regions:
        region_key = int(region.id[1:])
        region_id_to_instance_id[region_key] = []
        for instance in region.objects:
            instance_id = int(instance.id.split("_")[-1])
            region_id_to_instance_id[region_key].append(instance_id)
    return region_id_to_instance_id

def get_instance_id_to_all_dict(semantic_scene, save_explicit_dict=False):
    instance_id_to_all = {}
    for instance in semantic_scene.objects:
        instance_id = int(instance.id.split("_")[-1])
        if save_explicit_dict:
            instance = {
                "category_name": instance.category.name(),
                "category_index": instance.category.index(),
                "id": instance.id,
                "semantic_id": instance.semantic_id,
                "obb_center": instance.obb.center,
                "obb_sizes": instance.obb.sizes,
                "obb_rotation": instance.obb.rotation,
                "obb_world_to_local": instance.obb.world_to_local,
                "obb_local_to_world": instance.obb.local_to_world,
            }
        instance_id_to_all[instance_id] = instance
    return instance_id_to_all

def sample_goal_instances_across_regions(semantic_scene, seed=None):

    if seed is not None:
        np.random.seed(seed)

    cat_to_avoid = ['Unknown', 'wall', 'ceiling', 'floor']
    goal_instance_ids = []
    goal_instance_coords = []
    print("Num regions:", len(semantic_scene.regions))
    for region in semantic_scene.regions:

        # sample an instance not in cat_to_avoid
        instances_filtered = [insta for insta in region.objects if insta.category.name() not in cat_to_avoid]
        if len(instances_filtered) == 0:
            continue
        instance = np.random.choice(instances_filtered, replace=False)
        instance_coords = instance.aabb.center
        goal_instance_ids.append(obj_id_to_int(instance))
        goal_instance_coords.append(instance_coords)

        print(f"Region: {region.id}, Instance: {goal_instance_ids[-1]}, Category: {instance.category.name()}, coords: {instance_coords}, region center: {region.aabb.center}")

    return goal_instance_ids, goal_instance_coords

def sample_goal_instances_across_regions_indirect(semantic_scene, num_goals=2, repeat_regions=False):

    cat_to_avoid = ['Unknown', 'wall', 'ceiling', 'floor']
    reg_to_insta_dict = get_region_id_to_instance_id_dict(semantic_scene)
    insta_to_cat_map = get_instance_index_to_name_mapping(semantic_scene)
    insta_to_all_dict = get_instance_id_to_all_dict(semantic_scene)

    # sample regions
    num_extra_samples = 5 # to avoid regions with no filtered instances
    reg_ids = list(reg_to_insta_dict.keys())
    num_regions_to_sample = min(num_goals + num_extra_samples, len(reg_ids))
    reg_ids = np.random.choice(reg_ids, num_regions_to_sample, replace=repeat_regions)

    goal_instance_ids = []
    goal_instance_coords = []
    i = -1
    while len(goal_instance_ids) < num_goals:
        i += 1
        reg_id = reg_ids[i]

        # sample an instance not in cat_to_avoid
        insta_ids = reg_to_insta_dict[reg_id]
        insta_ids_filtered = [insta_id for insta_id in insta_ids if insta_to_cat_map[insta_id][1] not in cat_to_avoid]
        if len(insta_ids_filtered) == 0:
            continue
        insta_id = np.random.choice(insta_ids_filtered)
        insta_coords = insta_to_all_dict[insta_id].aabb.center
        goal_instance_ids.append(insta_id)
        goal_instance_coords.append(insta_coords)

        print(f"Region: {reg_id}, Instance: {insta_id}, Category: {insta_to_cat_map[insta_id][1]}, coords: {insta_coords}, region center: {semantic_scene.regions[reg_id].aabb.center}")

    return goal_instance_ids, goal_instance_coords


def obs_from_state(episode, state, sensor="color_sensor"):
    episode.agent.set_state(state)
    observations = episode.sim.get_sensor_observations()
    if sensor == "color_sensor":
        obs = np.array(Image.fromarray(observations["color_sensor"], mode="RGBA").convert('RGB'))
    else:
        obs = observations[sensor]
    return obs


def getImg(sim):
    observations = sim.get_sensor_observations()
    rgb = observations["color_sensor"]
    depth = observations["depth_sensor"]
    semantic = None
    if "semantic_sensor" in observations:
        semantic = observations["semantic_sensor"]
    return rgb, depth, semantic


def get_hm3d_scene_name_from_episode_path(path_episode, path_scenes_root_hm3d):
    episode_name = path_episode.parts[-1].split('_')[0]
    path_scene_hm3d = sorted(path_scenes_root_hm3d.glob(f'*{episode_name}'))[0]
    scene_name_hm3d = str(sorted(path_scene_hm3d.glob('*basis.glb'))[0])
    return scene_name_hm3d


def create_results_summary(args, results_summary, path_results_folder):
    # Calculate success rate
    results_summary['success_rate'] = (results_summary['successful_episodes'] /
                                       results_summary['total_episodes']) * 100 if results_summary[
                                                                                       'total_episodes'] > 0 else 0

    # Print and save results summary
    print("\n--- Results Summary ---")
    print(f"Total Episodes: {results_summary['total_episodes']}")
    print(f"Successful Episodes: {results_summary['successful_episodes']}")
    print(f"Failed Episodes: {results_summary['failed_episodes']}")
    print(f"Success Rate: {results_summary['success_rate']:.2f}%")

    print("\nFailure Reasons:")
    for reason, count in results_summary['failure_reasons'].items():
        print(f"  {reason}: {count}")

    # Save results to CSV
    if args.log_robot:
        results_csv_path = path_results_folder / 'results_summary.csv'
        with open(results_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Metric', 'Value'])
            csvwriter.writerow(
                ['Total Episodes', results_summary['total_episodes']])
            csvwriter.writerow(
                ['Successful Episodes', results_summary['successful_episodes']])
            csvwriter.writerow(
                ['Failed Episodes', results_summary['failed_episodes']])
            csvwriter.writerow(
                ['Success Rate (%)', f"{results_summary['success_rate']:.2f}"])

            csvwriter.writerow([])
            csvwriter.writerow(['Failure Reasons'])
            for reason, count in results_summary['failure_reasons'].items():
                csvwriter.writerow([reason, count])

    return results_summary