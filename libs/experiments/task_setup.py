import os
import numpy as np
from pathlib import Path
import pickle
import yaml
import torch
import cv2
from datetime import datetime
import shutil
from natsort import natsorted
import wandb
import networkx as nx

import habitat_sim

import logging

logger = logging.getLogger(
    "[Task Setup]"
)  # logger level is explicitly set below by LOG_LEVEL
from libs.logger.level import LOG_LEVEL

logger.setLevel(LOG_LEVEL)

from libs.goal_generator import goal_gen
from libs.experiments import model_loader
from libs.control.robohop import control_with_mask

from libs.common import utils_data
from libs.common import utils_visualize as utils_viz
from libs.common import utils_goals
from libs.common import utils
from libs.common import utils_sim_traj as ust
from libs.logger.visualizer import Visualizer


class Episode:
    def __init__(
        self, args, path_episode, scene_name_hm3d, path_results_folder, preload_data={}
    ):
        if args is None:
            args = utils.get_default_args()
        self.args = args
        self.steps = 0  # only used when running real in remote mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path_episode = path_episode
        logger.info(f"Running {self.path_episode=}...")
        self.scene_name_hm3d = scene_name_hm3d
        self.preload_data = preload_data

        self.init_controller_params()
        self.agent_states = None
        self.final_goal_position = None
        self.traversable_class_indices = None

        if self.args.log_robot:
            episode_results_dir = f"{self.path_episode.parts[-1]}_{self.args.method.lower()}_{self.args.goal_source}"
            self.path_episode_results = path_results_folder / episode_results_dir
            self.path_episode_results.mkdir(exist_ok=True, parents=True)
            self.set_logging()

        if args.env == "sim":
            if not (self.path_episode / "agent_states.npy").exists():
                raise FileNotFoundError(
                    f'{self.path_episode / "agent_states.npy"} does not exist...'
                )

        self.num_map_images = len(os.listdir(self.path_episode / "images"))
        self.cull_categories = [
            "floor",
            "floor mat",
            "floor vent",
            "carpet",
            "rug",
            "doormat",
            "shower floor",
            "pavement",
            "ground",
            "ceiling",
            "ceiling lower",
        ]

        # map params
        self.get_map_graph_path()
        self.graph_instance_ids, self.graph_path_lengths = None, None
        self.final_goal_mask_vis_sim = None

        # experiment params
        self.success_status = "exceeded_steps"
        self.distance_to_goal = np.nan
        self.step_real_complete = True

        self.image_height, self.image_width = (
            self.args.sim["height"],
            self.args.sim["width"],
        )
        self.sim, self.agent, self.distance_to_final_goal = None, None, np.nan
        if args.env == "sim":
            self.setup_sim_agent()
            self.ready_agent()
        self.load_map_graph()
        self.set_goal_generator()

        self.set_controller()

        # setup visualizer
        self.vis_img_default = np.zeros(
            (self.image_height, self.image_width, 3)
        ).astype(np.uint8)
        self.vis_img = self.vis_img_default.copy()
        self.video_cfg = {
            "savepath": str(self.path_episode_results / "repeat.mp4"),
            "codec": "mp4v",
            "fps": 6,
        }
        self.vis = Visualizer(
            self.sim, self.agent, self.scene_name_hm3d, env=self.args.env
        )
        if self.args.env == "sim":
            self.vis.draw_teach_run(self.agent_states)

            if self.args.task_type in ["alt_goal", "alt_goal_v2"]:
                self.vis.draw_goal(self.vis.sim_to_tdv(self.final_goal_position))

            if self.args.reverse:
                self.vis.draw_goal(self.vis.sim_to_tdv(self.final_goal_position))
                self.vis.draw_start(self.vis.sim_to_tdv(self.start_position))

    def init_controller_params(self):
        # data params
        # TODO: multiple hfov variables
        is_robohop_tango = np.isin(["robohop", "tango"], self.args.method.lower()).any()
        self.fov_deg = self.args.sim["hfov"] if is_robohop_tango else 79
        self.hfov_radians = np.pi * self.fov_deg / 180

        # controller params
        self.time_delta = 0.1
        self.theta_control = np.nan
        self.velocity_control = 0.05 if is_robohop_tango else np.nan
        self.pid_steer_values = (
            [0.25, 0, 0] if self.args.method.lower() == "tango" else []
        )
        self.discrete_action = -1
        self.controller_logs = None

    def setup_sim_agent(self) -> tuple:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

        # get the scene
        update_nav_mesh = False
        args_sim = self.args.sim
        self.sim, self.agent, vel_control = utils.get_sim_agent(
            self.scene_name_hm3d,
            update_nav_mesh,
            width=args_sim["width"],
            height=args_sim["height"],
            hfov=args_sim["hfov"],
            sensor_height=args_sim["sensor_height"],
        )
        self.sim.agents[0].agent_config.sensor_specifications[1].normalize_depth = True

        # create and configure a new VelocityControl structure
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True
        self.vel_control = vel_control

        (
            self.traversable_class_indices,
            self.bad_goal_classes,
            self.cull_instance_ids,
        ) = get_semantic_filters(
            self.sim, self.args.traversable_class_names, self.cull_categories
        )

    def ready_agent(self):
        # get the initial agent state for this episode (i.e. the starting pose)
        path_agent_states = self.path_episode / "agent_states.npy"
        self.agent_states = np.load(str(path_agent_states), allow_pickle=True)

        self.agent_positions_in_map = np.array([s.position for s in self.agent_states])

        # set the final goal state for this episode
        self.final_goal_state = None
        self.final_goal_position = None
        self.final_goal_image_idx = None
        if self.args.reverse:
            self.final_goal_state = utils_data.get_final_goal_state_reverse(
                self.sim, self.agent_states
            )
            self.final_goal_position = self.final_goal_state.position

            # set default, as it is not well defined for the reverse setting
            self.final_goal_image_idx = len(self.agent_states) - 1

        elif self.args.task_type in ["alt_goal", "alt_goal_v2"]:
            self.final_goal_image_idx, _, goal_instance_id = (
                utils_data.get_goal_info_alt_goal(
                    self.path_episode, self.args.task_type
                )
            )
            instance_position = None
            for instance in self.sim.semantic_scene.objects:
                if instance.semantic_id == goal_instance_id:
                    instance_position = instance.aabb.center
                    break
            if instance_position is None:
                raise ValueError("Could not obtain goal instance...")
            avg_floor_height = self.agent_positions_in_map[:, 1].mean()
            instance_position[1] = avg_floor_height
            self.final_goal_position = self.sim.pathfinder.snap_point(instance_position)
            self.agent_positions_in_map = self.agent_positions_in_map[
                : self.final_goal_image_idx + 1
            ]

        else:
            self.final_goal_position = self.agent_states[-1].position
            self.final_goal_image_idx = len(self.agent_states) - 1

        # set the start state and set the agent to this pose
        start_state = select_starting_state(
            self.sim, self.args, self.agent_states, self.final_goal_position
        )
        if start_state is None:
            self.success_status = (
                f"Could not find a valid start state for {self.path_episode}"
            )
            self.close()
            raise ValueError(self.success_status)
        self.agent.set_state(start_state)  # set robot to this pose
        self.start_position = start_state.position

        # define measure of success
        self.distance_to_final_goal = ust.find_shortest_path(
            self.sim, p1=self.start_position, p2=self.final_goal_position
        )[0]

    def get_map_graph_path(self):
        self.path_graph = None
        goal_source = self.args.goal_source.lower()
        if goal_source == "gt_metric":
            pass
        elif goal_source in ["gt_topological", "topological", "gt_topometric"]:
            # load robohop graph
            graph_filename = None
            if self.args.graph_filename is not None:
                graph_filename = self.args.graph_filename
            elif goal_source == "topological":
                suffix_str_depth = ""
                if self.args.goal_gen["edge_weight_str"] in [
                    "e3d_max",
                    "e3d_avg",
                    "e3d_min",
                ]:
                    suffix_str_depth = f"_depth_inferred"
                graph_filename = f'nodes_{self.args.goal_gen["map_segmentor_name"]}_graphObject_4_{self.args.goal_gen["map_matcher_name"]}{suffix_str_depth}.pickle'
            elif goal_source == "gt_topological":
                graph_filename = "nodes_graphObject_4.pickle"
            elif goal_source == "gt_topometric":
                graph_filename = "nodes_graphObject_4_gt_topometric.pickle"

            self.path_graph = self.path_episode / graph_filename
            if not self.path_graph.exists():
                raise FileNotFoundError(f"{self.path_graph} does not exist...")

    def load_map_graph(self):
        self.map_graph = None
        if self.path_graph is not None:
            logger.info(f"Loading graph: {self.path_graph}")
            map_graph = pickle.load(open(str(self.path_graph), "rb"))
            map_graph = utils.change_edge_attr(map_graph)
            self.map_graph = map_graph

            if self.args.goal_source == "topological":
                goalNodeIdx = None
                if not self.args.goal_gen["goalNodeIdx"] or self.args.env == "sim":
                    goalNodeIdx = self.get_task_goalNodeIdx()

                if self.args.cull_map_instances:
                    assert not self.args.goal_gen["rewrite_graph_with_allPathLengths"]
                    save_path = (
                        self.path_episode
                        / f"images_cull_mask_{self.args.cull_map_method}"
                    )
                    save_path.mkdir(exist_ok=True, parents=True)
                    self.map_graph = self.cull_map_instances(goalNodeIdx, save_path)

    def get_task_goalNodeIdx(self):
        self.goal_object_id, goalNodeIdx = None, None

        if not self.args.reverse:
            nodeID_to_imgRegionIdx = np.array(
                [self.map_graph.nodes[node]["map"] for node in self.map_graph.nodes()]
            )
            goalNodeIdx, self.final_goal_mask_vis_sim = utils_data.get_goalNodeIdx(
                str(self.path_episode),
                self.map_graph,
                nodeID_to_imgRegionIdx,
                self.args.task_type,
                ret_final_goalMask_vis=True,
            )

            if self.final_goal_mask_vis_sim is not None:
                cv2.imwrite(
                    str(self.path_episode_results / "final_goal_mask_vis_sim.jpg"),
                    self.final_goal_mask_vis_sim,
                )

        # reverse mode goalNodeIdx 'inferring' only for sim
        elif "sim" in self.args.env:
            reverse_goal_path = f"{self.path_episode}/reverse_goal.npy"
            if os.path.exists(reverse_goal_path):
                self.goal_object_id = np.load(reverse_goal_path, allow_pickle=True)[()][
                    "instance_id"
                ]
                goalNodeIdx = utils_data.get_goalNodeIdx_reverse(
                    str(self.path_episode), self.map_graph, self.goal_object_id
                )

        if goalNodeIdx is None:
            self.success_status = f"Could not find goalNodeIdx for {self.path_episode}"
            self.close()
            raise ValueError(self.success_status)

        self.args.goal_gen.update({"goalNodeIdx": goalNodeIdx})

        return goalNodeIdx

    def cull_map_instances(self, goalNodeIdx, save_cull_mask_path=None):
        map_graph = self.map_graph
        cull_categories = ["floor", "ceiling"]
        method = self.args.cull_map_method

        if method == "fast_sam":
            img_dir = self.path_episode / "images"
            if self.args.segmentor == "fast_sam":
                fast_sam = self.preload_data["segmentor"]
            else:
                fast_sam = model_loader.get_segmentor(
                    "fast_sam", self.image_width, self.image_height, device="cuda"
                )
        else:
            img_dir = self.path_episode / "images_sem"

        img_paths = natsorted(img_dir.iterdir())
        assert len(img_paths) > 0, f"No images found in {img_dir}"

        cull_inds = []
        for si, img_path in enumerate(img_paths):

            cull_mask_path_i = str(save_cull_mask_path / f"{si:04d}.jpg")
            # Temporarily disable loading cull masks from disk
            if 0:  # os.path.exists(cull_mask_path_i):
                cull_mask = cv2.imread(cull_mask_path_i, cv2.IMREAD_GRAYSCALE).astype(
                    bool
                )
            else:
                if method == "fast_sam":
                    img = cv2.imread(str(img_path))[:, :, ::-1]
                    cull_mask = fast_sam.segment(
                        img,
                        retMaskAsDict=False,
                        textLabels=cull_categories,
                        textCulls=False,
                    )[0]
                    if cull_mask is None:
                        cull_mask = fast_sam.no_mask.copy()
                        logger.info(f"cull_mask is None for image idx {si}")
                        continue
                    cull_mask = cull_mask.sum(0).astype(bool)
                else:
                    sem_instance = np.load(str(img_path), allow_pickle=True)
                    cull_mask = np.sum(
                        sem_instance[None, ...]
                        == self.cull_instance_ids[:, None, None],
                        axis=0,
                    ).astype(bool)

                if save_cull_mask_path is not None:
                    cv2.imwrite(cull_mask_path_i, cull_mask.astype(np.uint8) * 255)

            areaThresh = np.ceil(0.001 * cull_mask.shape[0] * cull_mask.shape[1])

            nodeInds = np.array(
                [
                    n
                    for n in map_graph.nodes
                    if map_graph.nodes[n]["map"][0] == si and n != goalNodeIdx
                ]
            )

            local_cull_count = 0
            for n in nodeInds:
                graph_sem_instance = utils.rle_to_mask(
                    map_graph.nodes[n]["segmentation"]
                )
                if graph_sem_instance[cull_mask].sum() >= areaThresh:
                    cull_inds.append(n)
                    local_cull_count += 1
            logger.info(
                f"{local_cull_count}/{len(nodeInds)} nodes to cull for image idx {si}"
            )
        logger.info(
            f"Before culling: {map_graph.number_of_nodes()=}, {map_graph.number_of_edges()=}"
        )
        edges = np.concatenate([list(map_graph.edges(n)) for n in cull_inds]).tolist()
        logger.info(f"{len(edges)} edges to remove")
        map_graph.remove_edges_from(edges)
        logger.info(
            f"After culling: {map_graph.number_of_nodes()=}, {map_graph.number_of_edges()=}"
        )

        # remove precomputed path lengths as the graph is now different
        allPathLengths = map_graph.graph.get("allPathLengths", {})
        edge_weight_str = self.args.goal_gen["edge_weight_str"]
        if edge_weight_str in allPathLengths:
            allPathLengths.pop(edge_weight_str)
        return map_graph

    def set_goal_generator(self):
        goal_source = self.args.goal_source.lower()
        if goal_source == "topological":
            segmentor_name = self.args.segmentor.lower()

            self.segmentor = self.preload_data["segmentor"]
            cfg_goalie = self.args.goal_gen
            cfg_goalie.update({"use_gt_localization": self.args.use_gt_localization})
            if segmentor_name == "sam2":
                assert (
                    cfg_goalie["matcher_name"] == "sam2"
                ), "TODO: is other matcher implemented for this segmentor?"
                cfg_goalie.update({"sam2_tracker": self.segmentor})

            self.goalie = goal_gen.Goal_Gen(
                W=self.image_width,
                H=self.image_height,
                G=self.map_graph,
                map_path=str(self.path_episode),
                poses=self.agent_states,
                task_type=self.args.task_type,
                cfg=cfg_goalie,
            )

            goalNodeImg = self.goalie.visualize_goal_node()
            cv2.imwrite(
                str(self.path_episode_results / "final_goal_mask_vis_pred.jpg"),
                goalNodeImg[:, :, ::-1],
            )

            if self.args.use_gt_localization:
                self.goalie.localizer.localizedImgIdx, _ = self.get_GT_closest_map_img()
            if self.args.reverse:
                self.goalie.localizer.localizer_iter_ub = self.num_map_images

            # to save time over storage
            if (
                not self.goalie.planner_g.precomputed_allPathLengths_found
                and not self.goalie.planner_g.preplan_to_goals_only
                and cfg_goalie["rewrite_graph_with_allPathLengths"]
            ):
                logger.info("Rewritng graph with allPathLengths")
                allPathLengths = self.map_graph.graph.get("allPathLengths", {})
                allPathLengths.update(
                    {
                        cfg_goalie[
                            "edge_weight_str"
                        ]: self.goalie.planner_g.allPathLengths
                    }
                )
                self.map_graph.graph["allPathLengths"] = allPathLengths
                pickle.dump(self.map_graph, open(self.path_graph, "wb"))

        elif goal_source == "gt_metric":
            # map_graph is not needed
            pass
        elif goal_source in ["gt_topological", "gt_topometric"]:
            self.get_goal_object_id()
            self.precompute_graph_paths()
        elif goal_source == "image_topological":
            if self.args.method.lower() == "learnt":
                self.goalie = type("", (), {})()
                self.goalie.config = self.preload_data["goal_controller"].config
                self.goalie.map_images = []
                self.goalie.loc_radius = self.goalie.config["loc_radius"]
                img_paths = natsorted((self.path_episode / "images/").iterdir())
                img_paths = img_paths[: self.final_goal_image_idx + 1]
                for img_path in img_paths:
                    self.goalie.map_images.append(cv2.imread(str(img_path))[:, :, ::-1])
                self.goalie.goal_idx, _ = self.get_GT_closest_map_img()
                self.goalie.num_map_images = len(self.goalie.map_images)
        else:
            raise NotImplementedError(f"{self.args.goal_source=} is not defined...")

    def get_goal_object_id(self):
        if self.args.reverse:
            self.goal_object_id = utils_data.find_reverse_traverse_goal(
                self.agent, self.sim, self.final_goal_state, self.map_graph
            )
            if not os.path.exists(f"{self.path_episode}/reverse_goal.npy"):
                print(f"Saving reverse goal to {self.path_episode}/reverse_goal.npy")
                np.save(
                    f"{self.path_episode}/reverse_goal.npy",
                    {
                        "instance_id": self.goal_object_id,
                        "agent_state": self.final_goal_state,
                    },
                )

        elif self.args.task_type in ["alt_goal", "alt_goal_v2"]:
            self.goal_object_id = utils_data.get_goal_info_alt_goal(
                self.path_episode, self.args.task_type
            )[-1]
        else:
            self.goal_object_id = int(str(self.path_episode).split("_")[-2])

    def precompute_graph_paths(self):
        self.graph_instance_ids, self.graph_path_lengths = (
            utils_goals.find_graph_instance_ids_and_path_lengths(
                self.map_graph,
                self.goal_object_id,
                device=self.device,
                weight=(
                    self.args.goal_gen["edge_weight_str"]
                    if self.args.goal_source.lower() == "gt_topometric"
                    else "margin"
                ),
            )
        )

    def set_controller(self):
        control_method = self.args.method.lower()
        goal_controller = None
        self.collided = None

        # select the type of controller to use
        if control_method == "tango":
            from libs.control.tango.pid import SteerPID
            from libs.control.tango.tango import TangoControl

            pid_steer = SteerPID(
                Kp=self.pid_steer_values[0],
                Ki=self.pid_steer_values[1],
                Kd=self.pid_steer_values[2],
            )

            intrinsics = utils.build_intrinsics(
                image_width=self.image_width,
                image_height=self.image_height,
                field_of_view_radians_u=self.hfov_radians,
                device=self.device,
            )

            goal_controller = TangoControl(
                traversable_classes=self.traversable_class_indices,
                pid_steer=pid_steer,
                default_velocity_control=self.velocity_control,
                h_image=self.image_height,
                w_image=self.image_width,
                intrinsics=intrinsics,
                time_delta=self.time_delta,
                grid_size=0.125,
                device=self.device,
            )

        elif control_method == "pixnav":
            from libs.pixnav.policy_agent import Policy_Agent
            from libs.pixnav.constants import POLICY_CHECKPOINT

            goal_controller = Policy_Agent(model_path=POLICY_CHECKPOINT)
            self.collided = False

        elif control_method == "learnt":
            goal_controller = self.preload_data["goal_controller"]
            goal_controller.reset_params()
            goal_controller.dirname_vis_episode = self.dirname_vis_episode

        self.goal_controller = goal_controller

    def get_GT_closest_map_img(self):
        dists = np.linalg.norm(
            self.agent_positions_in_map - self.agent.get_state().position, axis=1
        )
        topK = 2 * self.args.goal_gen["loc_radius"]
        closest_idxs = np.argsort(dists)[:topK]
        # approximately subsample ref indices
        closest_idxs = sorted(closest_idxs)[:: self.args.goal_gen["subsample_ref"]]
        closest_idx = np.argmin(dists)
        return closest_idx, closest_idxs

    def get_goal(self, rgb, depth, semantic_instance):
        goal_source = self.args.goal_source.lower()
        control_method = self.args.method.lower()
        self.traversable_mask = None
        self.goal_mask = None
        self.semantic_instance_predicted = None

        goal_img_idx, localizedImgInds = None, None
        if self.args.use_gt_localization:
            goal_img_idx, localizedImgInds = self.get_GT_closest_map_img()

        if goal_source == "gt_metric":
            _, plsDict, self.goal_mask = ust.get_pathlength_GT(
                self.sim,
                self.agent,
                depth,
                semantic_instance,
                self.final_goal_position,
                None,
            )
            self.control_input_robohop = semantic_instance

            instaIds, pls = list(zip(*plsDict.items()))
            masks = semantic_instance[None, ...] == np.array(instaIds)[:, None, None]
            self.control_input_learnt = [masks, np.array(pls)]

        elif goal_source in ["gt_topological", "gt_topometric"]:
            self.goal_mask = utils_goals.get_goal_mask_GT(
                graph_instance_ids=self.graph_instance_ids,
                pls=self.graph_path_lengths,
                sem=semantic_instance,
                device=self.device,
            )
            self.control_input_robohop = semantic_instance

            if not control_method == "learnt":
                # remove the naughty masks
                self.goal_mask[np.isin(semantic_instance, self.bad_goal_classes)] = 99

            masks = (
                semantic_instance[None, ...]
                == np.unique(semantic_instance)[:, None, None]
            )
            pls = [self.goal_mask[m].mean() for m in masks]
            self.control_input_learnt = [masks, np.array(pls)]

        elif goal_source == "topological":
            remove_mask = None
            # if 0:
            #     instance_ids_to_remove = np.concatenate([bad_goal_classes, traversable_class_indices])
            #     remove_mask = (semantic_instance_sim[:, :, None] == instance_ids_to_remove[None, None, :]).sum(-1).astype(bool)
            if len(self.args.goal_gen["textLabels"]) > 0:
                assert self.args.segmentor == "fast_sam"
            seg_results = self.segmentor.segment(
                rgb[:, :, :3], textLabels=self.args.goal_gen["textLabels"]
            )
            if self.args.segmentor.lower() in ["fast_sam", "sam2"]:
                self.semantic_instance_predicted, _, self.traversable_mask = seg_results
            else:
                self.semantic_instance_predicted = seg_results

            if self.args.cull_qry_instances:
                cull_mask = np.sum(
                    semantic_instance[None, ...]
                    == self.cull_instance_ids[:, None, None],
                    axis=0,
                ).astype(bool)
                qryMasks = utils.nodes2key(
                    self.semantic_instance_predicted, "segmentation"
                )
                areaThresh = np.ceil(
                    0.001 * semantic_instance.shape[0] * semantic_instance.shape[1]
                )
                cull_inds = []
                for mi, mask in enumerate(qryMasks):
                    # or mask.sum() <= areaThresh:
                    if mask[cull_mask].sum() >= areaThresh:
                        cull_inds.append(mi)
                logger.info(f"{len(cull_inds)}/{len(qryMasks)} instances to cull")
                if len(cull_inds) == len(qryMasks):
                    logger.warning("Skipped culling as len(cull_inds) == len(qryMasks)")
                else:
                    self.semantic_instance_predicted = [
                        self.semantic_instance_predicted[mi]
                        for mi in range(len(self.semantic_instance_predicted))
                        if mi not in cull_inds
                    ]

            if self.args.use_gt_localization:
                self.goalie.localizer.localizedImgIdx = (
                    goal_img_idx - 1 if self.args.reverse else goal_img_idx + 1
                )
                self.goalie.localizer.lost = False

            self.goal_mask = self.goalie.get_goal_mask(
                qryImg=rgb[:, :, :3],
                qryNodes=self.semantic_instance_predicted,
                qryPosition=(
                    self.agent.get_state().position
                    if self.args.debug and self.args.env == "sim"
                    else None
                ),
                remove_mask=remove_mask,
                refImgInds=localizedImgInds,
            )
            self.control_input_robohop = [self.goalie.pls, self.goalie.coords]
            self.control_input_learnt = [
                # self.goalie.qryMasks[self.goalie.matchPairs[:, 0]], self.goalie.pls]
                self.goalie.qryMasks,
                self.goalie.pls_min,
            ]
            if (
                self.goalie.qryMasks is not None
                and self.control_input_learnt[0] is not None
                and self.control_input_learnt[1] is not None
            ):
                assert len(self.control_input_learnt[0]) == len(
                    self.control_input_learnt[1]
                )

        elif goal_source == 'image_topological':
            if control_method == 'learnt':
                plan_shift = 1
                if self.args.use_gt_localization:
                    if self.args.reverse:
                        plan_shift = -1
                    self.goalie.goal_idx = goal_img_idx + plan_shift

                if self.args.use_gt_localization and self.goalie.config['fixed_plan']:
                    img_goal = self.goalie.map_images[min(
                        self.goalie.goal_idx, self.goalie.num_map_images - 1)]
                else:
                    start = max(self.goalie.goal_idx -
                                self.goalie.loc_radius, 0)
                    end = min(
                        self.goalie.goal_idx + self.goalie.loc_radius + 1, self.goalie.num_map_images)
                    img_goal_list = self.goalie.map_images[start:end]
                    self.goalie.goal_idx = self.goal_controller.predict_goal_idx(
                        rgb, img_goal_list, self.args.reverse)
                    img_goal = img_goal_list[self.goalie.goal_idx]
                    self.goalie.goal_idx += start
                self.control_input_learnt = img_goal
                self.goal_mask = img_goal.copy()
            else:
                raise NotImplementedError(
                    f'{goal_source=} only defined for {control_method=}...')

        else:
            raise NotImplementedError(f"{self.args.goal_source} is not available...")

    def get_control_signal(self, step, rgb, depth):
        control_method = self.args.method.lower()
        goals_image = None

        if control_method == "robohop":  # the og controller
            self.velocity_control, self.theta_control, goals_image = control_with_mask(
                self.control_input_robohop,
                self.goal_mask,
                v=self.velocity_control,
                gain=1,
                tao=5,
            )
            self.theta_control = -self.theta_control
            self.vis_img = (
                255.0
                - 255 * (utils_viz.goal_mask_to_vis(goals_image, outlier_min_val=255))
            ).astype(np.uint8)
        # the new and (hopefully) improved controller
        elif control_method == "tango":
            self.velocity_control, self.theta_control, goals_image_ = (
                self.goal_controller.control(
                    depth,
                    self.control_input_robohop,
                    self.goal_mask,
                    self.traversable_mask,
                )
            )
            if goals_image_ is not None:
                self.vis_img = (
                    255.0
                    - 255
                    * (utils_viz.goal_mask_to_vis(goals_image_, outlier_min_val=255))
                ).astype(np.uint8)
            else:
                self.vis_img = self.vis_img_default.copy()

        elif control_method == "pixnav":
            self.pixnav_goal_mask = utils.robohop_to_pixnav_goal_mask(
                self.goal_mask, depth
            )
            if not (step % 63) or self.discrete_action == 0:
                self.goal_controller.reset(rgb, self.pixnav_goal_mask.astype(np.uint8))
            self.discrete_action, predicted_mask = self.goal_controller.step(
                rgb, self.collided)
        elif control_method == 'learnt':
            if self.control_input_learnt[0] is None or self.control_input_learnt[1] is None:
                self.velocity_control, self.theta_control, self.vis_img = 0, 0, self.vis_img_default.copy()
            else:
                self.velocity_control, self.theta_control, self.vis_img = self.goal_controller.predict(
                    rgb, self.control_input_learnt)
            self.controller_logs = self.goal_controller.controller_logs
        else:
            raise NotImplementedError(f"{self.args.method} is not available...")
        return goals_image

    def execute_action(self):
        control_method = self.args.method.lower()

        if control_method == "pixnav":
            action_dict = {
                0: "stop",
                1: "move_forward",
                2: "turn_left",
                3: "turn_right",
                4: "look_up",
                5: "look_down",
            }
            previous_state = self.agent.state
            action = action_dict[self.discrete_action]
            _ = self.sim.step(action)
            current_state = self.agent.state
            self.collided = utils.has_collided(self.sim, previous_state, current_state)
        else:
            self.agent, self.sim, self.collided = utils.apply_velocity(
                vel_control=self.vel_control,
                agent=self.agent,
                sim=self.sim,
                velocity=self.velocity_control,
                steer=-self.theta_control,  # opposite y axis
                time_step=self.time_delta,
            )  # will add velocity control once steering is working

    def is_done(self):
        done = False
        current_robot_state = self.agent.get_state()  # world coordinates
        self.distance_to_goal = ust.find_shortest_path(
            self.sim, p1=current_robot_state.position, p2=self.final_goal_position
        )[0]
        if self.distance_to_goal <= self.args.threshold_goal_distance:
            logger.info(f"\nWinner! dist to goal: {self.distance_to_goal:.6f}\n")
            self.success_status = "success"
            done = True
        return done

    def set_logging(self):
        self.dirname_vis_episode = self.path_episode_results / "vis"
        self.dirname_vis_episode.mkdir(exist_ok=True, parents=True)

        self.filename_metadata_episode = self.path_episode_results / "metadata.txt"
        self.filename_results_episode = self.path_episode_results / "results.csv"

        utils.initialize_results(
            self.filename_metadata_episode,
            self.filename_results_episode,
            self.args,
            self.pid_steer_values,
            self.hfov_radians,
            self.time_delta,
            self.velocity_control,
            self.final_goal_position,
            self.traversable_class_indices,
        )

        results_dict_keys = [
            "step",
            "distance_to_goal",
            "velocity_control",
            "theta_control",
            "collided",
            "discrete_action",
            "agent_states",
            "controller_logs",
        ]
        self.results_dict = {k: [] for k in results_dict_keys}

    def log_results(self, step, final=False):
        if not final:
            utils.write_results(
                self.filename_results_episode,
                step,
                self.agent.get_state() if self.agent is not None else None,
                self.distance_to_goal,
                self.velocity_control,
                self.theta_control,
                self.collided,
                self.discrete_action,
            )
            if self.vis is not None:
                if self.args.env == "sim":
                    self.update_vis_sim()
                else:
                    self.update_vis()

            results_dict_curr = {
                "step": step,
                "distance_to_goal": self.distance_to_goal,
                "velocity_control": self.velocity_control,
                "theta_control": self.theta_control,
                "collided": self.collided,
                "discrete_action": self.discrete_action,
                "agent_states": self.agent.get_state() if self.agent is not None else None,
                "controller_logs": self.controller_logs[-1] if self.controller_logs is not None and len(self.controller_logs) > 0 else None,
            }

            self.update_results_dict(results_dict_curr)

        else:
            utils.write_final_meta_results(
                filename_metadata_episode=self.filename_metadata_episode,
                success_status=self.success_status,
                final_distance=self.distance_to_goal,
                step=step,
                distance_to_final_goal=self.distance_to_final_goal,
            )

            np.savez(
                self.path_episode_results / "results_dict.npz", **self.results_dict
            )

    def update_results_dict(self, curr_dict):
        for k, v in curr_dict.items():
            self.results_dict[k].append(v)

    def update_vis_sim(self):
        # if this is the first call, init video
        ratio = self.vis_img.shape[1] / self.vis.tdv.shape[1]
        if self.vis.video is None:
            # resize tdv to match the rgb image
            self.tdv = cv2.resize(self.vis.tdv, dsize=None, fx=ratio, fy=ratio)
            self.video_cfg["width"] = self.vis_img.shape[1]
            self.video_cfg["height"] = self.vis_img.shape[0] + self.tdv.shape[0]
            self.vis.init_video(self.video_cfg)

        self.vis.draw_infer_step(self.agent.get_state())
        self.tdv = cv2.resize(self.vis.tdv, dsize=None, fx=ratio, fy=ratio)
        combined_img = np.concatenate((self.tdv, self.vis_img), axis=0)
        self.vis.save_video_frame(combined_img)

    def update_vis(self):
        # if this is the first call, init video
        if self.vis.video is None:
            self.video_cfg["width"] = self.vis_img.shape[1]
            self.video_cfg["height"] = self.vis_img.shape[0]
            self.vis.init_video(self.video_cfg)

        self.vis.save_video_frame(self.vis_img)

    def init_plotting(self):
        # TODO: better handle 'plt' (check with SP)
        import matplotlib
        import matplotlib.pyplot as plt

        # matplotlib.use('Qt5Agg')  # need it on my environment on my system ------
        if self.args.save_vis:
            matplotlib.use("Agg")  # Use the Agg backend to suppress plots

        import matplotlib.style as mplstyle

        mplstyle.use("fast")
        mplstyle.use(["dark_background", "ggplot", "fast"])
        fig, ax = utils_viz.setup_sim_plots()
        return ax, plt

    def plot(self, ax, plt, step, rgb, depth, semantic_instance):
        goals_image = None

        if self.args.goal_source.lower() == "topological":
            if self.semantic_instance_predicted is None:
                semantic_instance_vis = np.zeros(rgb.shape[:2])
            else:
                semantic_instance_vis = utils_viz.show_anns(
                    None, self.semantic_instance_predicted, borders=False
                )
        else:
            semantic_instance_vis = semantic_instance

        goal_mask_vis = utils_viz.goal_mask_to_vis(self.goal_mask)
        goal = self.goal_mask == self.goal_mask.min()  # .astype(int)
        if self.args.method.lower() == "pixnav":
            goal += (self.pixnav_goal_mask / self.pixnav_goal_mask.max()).astype(
                int
            ) * 2
        utils_viz.plot_sensors(
            ax=ax,
            display_img=rgb,
            semantic=semantic_instance_vis,
            depth=depth,
            goal=goal,
            goal_mask=goal_mask_vis,
            flow_goal=(
                goals_image if goals_image is not None else np.zeros(rgb.shape[:2])
            ),
            trav_mask=self.traversable_mask,
        )
        if self.args.method.lower() == "tango":
            utils_viz.plot_path_points(
                ax=[ax[1, 2], ax[1, 0]],
                points=self.goal_controller.point_poses,
                cost_map_relative_bev=self.goal_controller.planning_cost_map_relative_bev_safe,
                colour="red",
            )

        if self.args.save_vis:
            plt.tight_layout()
            plt.savefig(
                self.dirname_vis_episode / f"{step:04d}.jpg",
                bbox_inches="tight",
                pad_inches=0,
            )
        else:
            plt.pause(0.05)  # pause a bit so that plots are updated

    def close(self, step=-1):
        # if self.args.plot:
        # plt.close()
        if self.args.log_robot:
            self.log_results(step, final=True)

        if hasattr(self, "vis") and self.vis:
            self.vis.close()
        if hasattr(self, "sim") and self.sim:
            self.sim.close()


def setup_wandb_logging(args):
    wandb.login()
    wandb.init(project="obj_rel_nav")
    wandb.config.update(args)
    wandb.run.name = (
        f"{args.exp_name}_{args.task_type}_{args.method}_{args.goal_source}"
    )


def wandb_log_episode(epsiode_name, results_dict, video_path=None):
    for step in range(len(results_dict["step"])):
        wandb.log(
            {
                f"{epsiode_name}/{key}": results_dict[key][step]
                for key in results_dict.keys()
                if key not in ["step", "agent_states"]
            },
            commit=False,
        )
    wandb.log({})

    if video_path is not None and os.path.exists(video_path):
        video_path_2 = video_path[:-4] + "_2.mp4"
        os.system(
            f"ffmpeg -y -loglevel 0 -i {video_path} -vcodec libx264 {video_path_2}"
        )
        wandb.log({"video": wandb.Video(video_path_2)}, commit=True)


def load_run_list(args, path_episode_root) -> list:
    if args.run_list == "":
        path_episodes = sorted(path_episode_root.glob("*"))
    else:
        path_episodes = []
        if args.path_run == "":
            raise ValueError("Run path must be specified when using run list!")
        if args.run_list.lower() in ["winners", "failures", "no_good", "custom"]:
            if args.run_list.lower() not in ["no_good", "custom"]:
                logger.info(
                    f"Setting logging to False when running winner or failure list! - arg.log_robot:{args.log_robot}"
                )
                args.log_robot = False
            with open(
                str(Path(args.path_run) / "summary" / f"{args.run_list.lower()}.csv"),
                "r",
            ) as f:
                for line in f.readlines():
                    path_episodes.append(
                        path_episode_root
                        / line[: line.rfind(f"_{args.method}")].strip("\n")
                    )
        else:
            raise ValueError(f"{args.run_list} is not a valid option.")
    return path_episodes


def init_results_dir_and_save_cfg(args, default_logger=None):
    if (args.log_robot or args.save_vis) and args.run_list == "":
        path_results = Path(args.path_results)
        task_str = args.task_type
        if args.reverse:
            task_str += "_reverse"
        path_results_folder = (
            path_results
            / task_str
            / args.exp_name
            / args.split
            / args.max_start_distance
            / f'{datetime.now().strftime("%Y%m%d-%H-%M-%S")}_{args.method.lower()}_{args.goal_source}'
        )
        path_results_folder.mkdir(exist_ok=True, parents=True)
        if default_logger is not None:
            default_logger.update_file_handler_root(path_results_folder / "output.log")
        print(f"Logging to: {str(path_results_folder)}")
    elif args.run_list != "":
        path_results_folder = Path(args.path_run)
        print(f"(Overwrite) Logging to: {str(path_results_folder)}")

    if args.log_robot:
        save_dict(path_results_folder / "args.yaml", vars(args))
        if args.method.lower() == "learnt":
            shutil.copyfile(
                args.controller["config_file"], path_results_folder / "gnm_test.yaml"
            )

    return path_results_folder


def preload_models(args):
    # preload some models before iterating over the episodes
    goal_controller = model_loader.get_controller_model(
        args.method, args.goal_source, args.controller["config_file"])

    segmentor = None
    if args.goal_source == "topological":

        # use predefined traversable classes with fast_sam predictions only if it is tango and infer_traversable is True
        traversable_class_names = (
            args.traversable_class_names
            if args.method.lower() == "tango" and args.infer_traversable
            else None
        )

        segmentor = model_loader.get_segmentor(
            args.segmentor,
            args.sim["width"],
            args.sim["height"],
            path_models=args.path_models,
            traversable_class_names=traversable_class_names,
        )

    depth_model = None
    if args.infer_depth:
        depth_model = model_loader.get_depth_model()

    # collect preload data that each episode instance can reuse
    preload_data = {
        "goal_controller": goal_controller,
        "segmentor": segmentor,
        "depth_model": depth_model,
    }
    return preload_data


def set_start_state_reverse_orientation(agent_states, start_index):
    start_state = agent_states[start_index]
    # compute orientation, looking at the next GT forward step
    lookat_index = start_index - 1
    if lookat_index < 0:
        print("Cannot reverse orientation at the start of the episode.")
        return None

    # search/validate end_idx in reverse direction
    for k in range(lookat_index, -1, -1):
        # keep looking if agent hasn't moved
        if np.linalg.norm(start_state.position - agent_states[k].position) <= 0.1:
            continue
        else:
            lookat_index = k
            break
    # looking in the reverse direction
    start_state.rotation = ust.get_agent_rotation_from_two_positions(
        start_state.position, agent_states[lookat_index].position
    )
    return start_state


def closest_state(sim, agent_states, distance_threshold: float, final_position=None):
    distances = np.zeros_like(agent_states)
    final_position = (
        agent_states[-1].position if final_position is None else final_position
    )
    for i, p in enumerate(agent_states):
        distances[i] = ust.find_shortest_path(sim, final_position, p.position)[0]
    start_index = ((distances - distance_threshold) ** 2).argmin()
    return start_index


def select_starting_state(sim, args, agent_states, final_position=None):
    # reverse traverse episodes end 1m before the original start, offset that
    distance_threshold_offset = 1 if args.reverse else 0
    if args.max_start_distance.lower() == "easy":
        start_index = closest_state(
            sim, agent_states, 3 + distance_threshold_offset, final_position
        )
    elif args.max_start_distance.lower() == "hard":
        if args.task_type == "via_alt_goal":
            distance_threshold_offset += 3

        start_index = closest_state(
            sim, agent_states, 5 + distance_threshold_offset, final_position
        )
    elif args.max_start_distance.lower() == "full":
        start_index = 0 if not args.reverse else len(agent_states) - 1
    else:
        raise NotImplementedError(
            f"max start distance: {args.max_start_distance} is not an available start."
        )
    start_state = agent_states[start_index]
    if args.reverse:
        start_state = set_start_state_reverse_orientation(agent_states, start_index)
    return start_state


def save_dict(full_save_path, config_dict):
    with open(full_save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_semantic_filters(sim, traversable_class_names, cull_categories):
    # setup is/is not traversable and which goals are banned (for the simulator runs)
    instance_index_to_name_map = utils.get_instance_index_to_name_mapping(
        sim.semantic_scene
    )
    traversable_class_indices = instance_index_to_name_map[:, 0][
        np.isin(instance_index_to_name_map[:, 1], traversable_class_names)
    ]
    traversable_class_indices = np.unique(traversable_class_indices).astype(int)
    bad_goal_categories = ["ceiling", "ceiling lower"]
    bad_goal_cat_idx = instance_index_to_name_map[:, 0][
        np.isin(instance_index_to_name_map[:, 1], bad_goal_categories)
    ]
    bad_goal_classes = np.unique(bad_goal_cat_idx).astype(int)

    cull_instance_ids = (
        instance_index_to_name_map[:, 0][
            np.isin(instance_index_to_name_map[:, 1], cull_categories)
        ]
    ).astype(int)
    return traversable_class_indices, bad_goal_classes, cull_instance_ids
