import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional


def setup_sim_plots():
    fig, ax = plt.subplots(2, 4, figsize=(12, 7), constrained_layout=True)
    return fig, ax


def value_to_color(values, vmin=None, vmax=None, cm_name="jet"):
    # check if colormaps has get_cmap method (newer versions of matplotlib)
    if hasattr(matplotlib.colormaps, "get_cmap"):
        cmap_paths = matplotlib.colormaps.get_cmap(cm_name)
    else:
        cmap_paths = matplotlib.cm.get_cmap(cm_name)
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = np.array([cmap_paths(norm(value))[:3] for value in values])
    return colors, norm


def visualize_flow(
    cords_org,
    cords_dst,
    img=None,
    colors=None,
    norm=None,
    weights=None,
    cmap="jet",
    colorbar=True,
    display=True,
    fwdVals=None,
):

    diff = cords_org - cords_dst
    dpi = 100
    img_height, img_width = img.shape[:2]  # Get the image dimensions
    fig_width, fig_height = img_width / dpi, img_height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    if img is not None:
        ax.imshow(img)
    # for i in range(len(currImg_mask_coords)):
    #     ax.plot(currImg_mask_coords[i,0],currImg_mask_coords[i,1],'o',color='r')
    #     ax.plot(refNodes_mask_coords[matchInds[i],0],refNodes_mask_coords[matchInds[i],1],'o',color='b')
    if fwdVals is not None:
        # plot a diamond for negative values and a circle for positive values, size = val
        pointTypeMask = fwdVals > 0
        ax.scatter(
            *(cords_org[pointTypeMask].T),
            c=colors[pointTypeMask],
            s=abs(fwdVals[pointTypeMask]) * 40,
            marker="o",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.scatter(
            *(cords_org[~pointTypeMask].T),
            c=colors[~pointTypeMask],
            s=abs(abs(fwdVals[~pointTypeMask])) * 40,
            marker="X",
            edgecolor="white",
            linewidth=0.5,
        )
    if weights is not None:
        weighted_sum = (weights[:, None] * diff).sum(0)
        ax.quiver(
            *np.array([160, 120]).T,
            weighted_sum[0],
            weighted_sum[1],
            color="black",
            edgecolor="white",
            linewidth=0.5,
        )
    ax.quiver(
        *cords_org.T,
        diff[:, 0],
        diff[:, 1],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlim([0, img_width])
    ax.set_ylim([img_height, 0])
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if display:
        plt.show()
    else:
        # return the figure as image (same size as img imshow-ed above)
        fig.canvas.draw()
        vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        vis = cv2.resize(vis, (img.shape[1], img.shape[0]))
        plt.close(fig)
        return vis


def add_colobar(ax, plt, norm=None, cmap="jet"):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create a ScalarMappable object with the "autumn" colormap
    if norm is None:
        norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Add a colorbar to the axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")

    # Customize the colorbar
    cbar.set_label("Colorbar Label", labelpad=10)
    cbar.ax.yaxis.set_ticks_position("right")


def show_anns(ax, anns, borders=True, display=False):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    if ax is not None:
        ax = plt.gca()
        ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    if display and ax is not None:
        ax.imshow(img)
    return img


def draw_masks_with_colours(im_, masks, colors, alpha=0.5):
    im = im_.copy() / 255.0
    viz = im.copy()  # np.zeros((im.shape[0],im.shape[1],3))
    for j in range(len(masks)):
        viz[masks[j]] = colors[j]
    im = alpha * viz + (1 - alpha) * im
    return im


def goal_mask_to_vis(goal_mask, outlier_min_val=99):
    """
    convert goal mask to visualisation mask
    """
    goal_mask_vis = goal_mask.copy()
    outlier_mask = goal_mask_vis >= outlier_min_val
    # if all are outliers, set all to outlier_min_val
    if np.sum(~outlier_mask) == 0:
        goal_mask_vis = outlier_min_val * np.ones_like(goal_mask_vis)
    # elif it is a mix of inliers/outliers, replace outliers with max of inliers
    elif np.sum(outlier_mask) != 0 and np.sum(~outlier_mask) != 0:
        goal_mask_vis[outlier_mask] = goal_mask_vis[~outlier_mask].max() + 1
    # invert the mask for visualisation
    goal_mask_vis = goal_mask_vis.max() - goal_mask_vis
    return goal_mask_vis


def plot_sensors(
    ax,
    display_img: np.ndarray,
    semantic: np.ndarray,
    depth: np.ndarray,
    goal: Optional[np.ndarray] = None,
    goal_mask: Optional[np.ndarray] = None,
    flow_goal: Optional[np.ndarray] = None,
    trav_mask: Optional[np.ndarray] = None,
):
    ax[0, 0].cla()
    ax[0, 0].imshow(display_img)
    ax[0, 0].set_title(f"RGB")
    # Hide X and Y axes label marks
    ax[0, 0].xaxis.set_tick_params(labelbottom=False)
    ax[0, 0].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])

    ax[0, 1].cla()
    ax[0, 1].imshow(semantic)
    ax[0, 1].set_title("Segments")
    # Hide X and Y axes label marks
    ax[0, 1].xaxis.set_tick_params(labelbottom=False)
    ax[0, 1].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    ax[0, 2].cla()
    ax[0, 2].imshow(depth)
    ax[0, 2].set_title("Depth")
    # Hide X and Y axes label marks
    ax[0, 2].xaxis.set_tick_params(labelbottom=False)
    ax[0, 2].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    extent = [-6, 6, 0, 10]
    ax[1, 1].cla()
    ax[1, 1].imshow(display_img, extent=extent)
    ax[1, 1].imshow(trav_mask, alpha=0.5, extent=extent, cmap="Greens")
    ax[1, 1].xaxis.set_tick_params(labelbottom=False)
    ax[1, 1].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title("Traversability")
    ax_flow = ax[1, 2]
    ax_flow.cla()
    ax_flow.imshow(flow_goal, extent=extent, cmap="viridis")
    ax_flow.set_title("Fallback (RoboHop)")
    ax_flow.xaxis.set_tick_params(labelbottom=False)
    ax_flow.yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax_flow.set_xticks([])
    ax_flow.set_yticks([])

    if goal_mask is not None:
        # goal_mask = np.clip(goal_mask, 0, 19)
        ax[0, 3].imshow(display_img)
        ax[0, 3].imshow(goal_mask, alpha=0.5, cmap="viridis")
        ax[0, 3].xaxis.set_tick_params(labelbottom=False)
        ax[0, 3].yaxis.set_tick_params(labelleft=False)
        ax[0, 3].set_xticks([])
        ax[0, 3].set_yticks([])
        ax[0, 3].set_title("Goal Mask")
    goal_cmap = matplotlib.colors.ListedColormap(
        [matplotlib.cm.get_cmap("Greys")(0.0), matplotlib.cm.get_cmap("viridis")(1.0)]
    )
    ax[1, 3].cla()
    ax[1, 3].imshow(display_img, extent=extent)
    ax[1, 3].imshow(goal, alpha=0.5, extent=extent, cmap=goal_cmap)
    ax[1, 3].set_title("Selected Goal")
    ax[1, 3].xaxis.set_tick_params(labelbottom=False)
    ax[1, 3].yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax[1, 3].set_xticks([])
    ax[1, 3].set_yticks([])
    return ax


def plot_path_points(ax, points, cost_map_relative_bev, colour: str = "blue"):
    ax[1].cla()
    ax[1].imshow(
        cost_map_relative_bev, cmap="inferno", origin="lower", extent=[-6, 6, 0, 10]
    )
    ax[1].plot(points[:, 0], points[:, 1])
    ax[1].set_title("BEV")
