"""
Learnt controller for the robot
"""

import sys
import numpy as np
import yaml
import io
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend to suppress plots
import logging

# # TODO
# import warnings

# warnings.filterwarnings(
#     "ignore", "You are using `torch.load` with `weights_only=False`*."
# )
# warnings.filterwarnings("ignore", "Don't use ConvNormActivation directly, *")

# Add object_react folder to sys.path using absolute paths (needed to access relative imports)
from pathlib import Path

main_dir = Path(__file__).resolve().parents[2]
sys.path.append(f"{main_dir}/libs/control/object_react/train")

logger = logging.getLogger(
    "[Controller]"
)  # logger level is explicitly set below by LOG_LEVEL

from libs.control.object_react.train.vint_train.models.gnm.gnm import GNM
from libs.control.object_react.train.train import resume_model
from libs.control.object_react.train.vint_train.training.train_utils import (
    get_goal_image,
    get_obs_image,
)
from libs.control.object_react.train.vint_train.data.data_utils import (
    resize_and_aspect_crop,
)
from libs.control.object_react.train.vint_train.objectreact_dataloader import (
    generate_positional_encodings,
    normalize_pls,
    get_masks_gradient,
)
from libs.control.object_react.train.vint_train.visualizing.action_utils import (
    gen_bearings_from_waypoints,
)
from libs.logger.level import LOG_LEVEL

logger.setLevel(LOG_LEVEL)


class ObjRelLearntController:
    """
    Object relative learnt controller
    """

    def __init__(self, config, **kwargs):
        if type(config) == str:
            with open(config, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        elif type(config) == dict:
            self.config = config
        else:
            raise ValueError(f"config must be a filepath or a dict, not {type(config)}")

        self.goal_source = kwargs.get("goal_source", None)

        self.dirname_vis_episode = kwargs.get("dirname_vis_episode", None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.goal_type = self.config["goal_type"]
        self.obs_type = self.config["obs_type"]
        self.image_size = self.config["image_size"]
        if self.goal_source == "image_topological":
            assert (
                self.obs_type == "image"
            ), f"{self.goal_source=} requires obs_type=image"
            assert (
                self.goal_type == "image"
            ), f"{self.goal_source=} requires goal_type=image"
        else:
            assert (
                self.goal_type == "image_mask_enc"
            ), f"{self.goal_source=} requires goal_type=image_mask_enc"

        self.pl_outlier_value = 99
        if "e3d_" in self.config["load_run"]:
            self.pl_outlier_value = 255
        self.is_pl_normalized = self.config["is_pl_normalized"]
        self.use_vel_filter = self.config["use_vel_filter"]

        self.model = GNM(
            self.config["context_size"],
            self.config["len_traj_pred"],
            self.config["learn_angle"],
            self.config["obs_encoding_size"],
            self.config["goal_encoding_size"],
            goal_type=self.goal_type,
            obs_type=self.obs_type,
            dims=self.config["dims"],
            goal_uses_context=self.config["goal_uses_context"],
            use_mask_grad=self.config["use_mask_grad"],
            **kwargs,
        )

        _ = resume_model(
            self.config, self.model, load_project_folder=self.config["load_run"]
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(self.transform)

        # init params
        self.reset_params()

        self.rank_enc = generate_positional_encodings(200, self.config["dims"])
        self.waypoint_index = self.config["waypoint_index"]
        print("Learnt controller initialized!")

    def reset_params(self):
        """
        Reset the controller
        """
        self.iter = -1
        self.image_history = []
        self.goal_history = []
        self.action_history = []
        self.controller_logs = []

        self.goal_mask_vis = None
        self.action_pred = None

        self.v_rollout = np.zeros(self.config["len_traj_pred"])
        self.w_rollout = np.zeros(self.config["len_traj_pred"])

    def maintain_history(self, curr, history):
        """
        Maintain history for context
        """
        diff = len(history) - self.config["context_size"] - 1
        if diff < 0:
            for _ in range(abs(diff)):
                history.append(curr)
        else:
            history.pop(0)
            history.append(curr)

    def encode_goal(self, goal_data):
        """
        Encode goal data to feed to the model
        """
        masks, pls = goal_data
        if not self.is_pl_normalized:
            pls = normalize_pls(pls, outlier_value=self.pl_outlier_value)
        pls = pls.astype(np.uint8)
        masks = masks.transpose([1, 2, 0])[::4, ::4]  # (H,W,D)
        mask_vis = np.zeros((masks.shape[0], masks.shape[1]))
        for m in range(masks.shape[-1]):
            mask_vis[masks[:, :, m]] = pls[m]
        enc = self.rank_enc[pls.astype(int)]
        goal_enc = (masks @ enc).transpose(2, 0, 1)
        if self.config["use_mask_grad"]:
            grad = get_masks_gradient(masks.transpose(2, 0, 1))
            goal_enc = np.concatenate([goal_enc, grad[None]], axis=0)
        return goal_enc, mask_vis

    def filter_vel(self, action, win=5, self_update=True):
        self.action_history.append(action)
        if len(self.action_history) > win:
            self.action_history.pop(0)
        action = np.array(self.action_history).mean(axis=0)
        if self_update:
            self.action_history[-1] = action
        return action

    def waypoint_to_velocity(self, waypoint, time_step=1):
        """PD controller for the robot"""
        eps = 1e-8
        max_v, max_w = 0.2, 0.1

        assert (
            len(waypoint) == 2 or len(waypoint) == 4
        ), "waypoint must be a 2D or 4D vector"
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint
        # this controller only uses the predicted heading if dx and dy near zero
        if len(waypoint) == 4 and np.abs(dx) < eps and np.abs(dy) < eps:
            v = 0
            w = clip_angle(np.arctan2(hy, hx)) / time_step
        elif np.abs(dx) < eps:
            v = 0
            w = np.sign(dy) * np.pi / (2 * time_step)
        else:
            v = dx / time_step
            w = np.arctan(dy / dx) / time_step
        v = np.clip(v, 0, max_v)
        w = np.clip(w, -max_w, max_w)
        return v, w

    def ready_obs(self, rgb):
        obs_image = resize_and_aspect_crop(Image.fromarray(rgb), self.image_size)
        self.maintain_history(obs_image, self.image_history)
        obs_image = torch.as_tensor(torch.cat(self.image_history), dtype=torch.float32)
        obs_image, _ = get_obs_image(
            obs_image[None, ...], self.obs_type, self.transform, self.device
        )
        return obs_image

    def ready_goal(self, goal_data):
        if self.goal_type == "image":
            if self.config["goal_uses_context"]:
                raise NotImplementedError(
                    f"TODO: Implement context for {self.goal_type=}"
                )
            if type(goal_data) == np.ndarray:
                goal_data = [goal_data]
            goal_images = []
            for i in range(len(goal_data)):
                goal_image = resize_and_aspect_crop(
                    Image.fromarray(goal_data[i]), self.image_size
                )
                goal_images.append(goal_image)
            self.goal_mask_vis = goal_images[-1].cpu().numpy().transpose(1, 2, 0)
            goal_image = torch.stack(goal_images)
        else:
            goal_enc, self.goal_mask_vis = self.encode_goal(goal_data)

            if self.config["goal_uses_context"]:
                self.maintain_history(
                    torch.as_tensor(goal_enc, dtype=torch.float32), self.goal_history
                )
                goal_enc = torch.as_tensor(
                    torch.cat(self.goal_history), dtype=torch.float32
                )

            goal_vis = goal_enc[:3, :, :]  # temporary variable
            if self.config["dims"] < 3:
                goal_vis = goal_enc[:1].repeat(3, axis=0)
            goal_image = torch.as_tensor(
                np.concatenate([goal_vis, goal_enc], axis=0), dtype=torch.float32
            )[None, ...]

        goal_image, _ = get_goal_image(
            goal_image, self.goal_type, self.transform, self.device
        )
        return goal_image

    def predict_goal_idx(self, rgb, goal_data, reverse=False):
        with torch.no_grad():
            obs_image = self.ready_obs(rgb)
            goal_image = self.ready_goal(goal_data)
            obs_image = obs_image.repeat(goal_image.shape[0], 1, 1, 1)

            dist_pred, _ = self.model(obs_image, goal_image)  # (B, 1)
            dist_pred = dist_pred[:, 0].float().cpu().numpy()

        closest_goal_idx = np.argmin(dist_pred)
        plan_shift = 1
        if reverse:
            plan_shift = -1
        if dist_pred[closest_goal_idx] <= self.config["close_threshold"]:
            closest_goal_idx = max(
                0,
                min(
                    closest_goal_idx + plan_shift,
                    self.config["len_traj_pred"] - 1,
                    len(dist_pred) - 1,
                ),
            )
        return closest_goal_idx

    def predict(self, rgb, goal_data):
        """
        predict the linear velocity and angular velocity
        """
        self.iter += 1
        v, w = 0, 0
        with torch.no_grad():
            obs_image = self.ready_obs(rgb)
            goal_image = self.ready_goal(goal_data)

            model_outputs = self.model(obs_image, goal_image)
            _, action_pred = model_outputs
            self.action_pred = action_pred[0].float().cpu().numpy()
            wp = self.action_pred[self.waypoint_index][:2]

            w_rollout = np.arctan2(self.action_pred[:, 1], self.action_pred[:, 0])
            w_rollout = np.insert(w_rollout, 0, 0)
            self.w_rollout = -(w_rollout[1:] - w_rollout[:-1])
            v_rollout = 0.2 * self.action_pred[:, 0]
            v_rollout = np.insert(v_rollout, 0, 0)
            self.v_rollout = v_rollout[1:] - v_rollout[:-1]

            w = np.arctan2(wp[-1], wp[-2])
            w = np.clip(w, -0.1, 0.1)
            v = min(wp[0] / 100, 0.05)
            if self.use_vel_filter:
                v, w = self.filter_vel([v, w])

            logger.info(f"Predicted lin: {v:.2f} and ang: {w:.2f}")
            vis_img = visualize_prediction(
                rgb,
                self.action_pred,
                self.goal_mask_vis,
                save_path=None,
                get_plot_img=True,
            )

            logger.info("Visualization created")
            self.controller_logs.append(
                {
                    "action_pred": self.action_pred,
                    "v_rollout": self.v_rollout,
                    "w_rollout": self.w_rollout,
                }
            )
        return v, -w, vis_img


def visualize_prediction(
    rgb,
    pred_waypoints,
    goal_mask_vis=None,
    save_path=None,
    display=False,
    dpi=50,
    get_plot_img=False,
):
    """
    Visualize the prediction
    """
    fig, ax = plt.subplots(1, 3)
    plot_traj(ax[0], pred_waypoints)

    ax[1].imshow(rgb)
    ax[1].axis("off")

    w, h = 160, 120
    if goal_mask_vis is not None:
        w, h = goal_mask_vis.shape[0], goal_mask_vis.shape[1]
        ax[2].imshow(goal_mask_vis)
        ax[2].axis("off")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    if display:
        plt.show()

    image = 255 * np.ones([w, h, 3]).astype(np.uint8)
    if get_plot_img:
        to_img_method = "canvas"
        # Save the plot to an in-memory buffer
        if to_img_method == "savefig":
            with io.BytesIO() as buffer:
                plt.savefig(
                    buffer, format="jpg", bbox_inches="tight", pad_inches=0, dpi=dpi
                )
                buffer.seek(0)  # Rewind the buffer to the beginning
                image = np.array(Image.open(buffer))
        elif to_img_method == "canvas":
            fig.canvas.draw()  # Render the figure
            image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        else:
            pass

    plt.close()

    return image


def plot_traj(ax, traj, quiver_freq=1):
    """
    Plot trajectory
    """
    ax.plot(
        traj[:, 1],
        traj[:, 0],
        color="c",
        alpha=0.5,
        marker="o",
    )
    bearings = gen_bearings_from_waypoints(traj)
    ax.quiver(
        traj[::quiver_freq, 1],
        traj[::quiver_freq, 0],
        -bearings[::quiver_freq, 1],
        bearings[::quiver_freq, 0],
        color="y",
        scale=1.0,
    )
    ax.set_ylim(-1, 12)
    ax.set_xlim(-4, 4)
    ax.invert_xaxis()
    ax.set_aspect("equal", "box")


def clip_angle(theta) -> float:
    """Clip angle to [-pi, pi]"""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi
