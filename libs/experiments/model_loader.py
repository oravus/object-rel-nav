import torch
from pathlib import Path

# ignore FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="contextlib")


def get_depth_model():
    from libs.depth.depth_anything_metric_model import DepthAnythingMetricModel

    depth_model_name = 'zoedepth'
    path_zoe_depth = Path.cwd() / 'model_weights' / \
        'depth_anything_metric_depth_indoor.pt'
    if not path_zoe_depth.exists():
        raise FileNotFoundError(f'{path_zoe_depth} not found...')
    depth_model = DepthAnythingMetricModel(
        depth_model_name, pretrained_resource=str(path_zoe_depth))
    return depth_model


def get_controller_model(method, goal_source, config_filepath):
    goal_controller = None
    if method == 'learnt':
        from libs.control.objectreact import ObjRelLearntController
        goal_controller = ObjRelLearntController(
            config_filepath, goal_source=goal_source)
    return goal_controller


def get_segmentor(segmentor_name, image_width, image_height, device=None,
                  path_models=None, traversable_class_names=None):
    if device is None:
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

    segmentor = None

    if segmentor_name == 'sam':
        from libs.segmentor import sam

        segmentor = sam.Seg_SAM(
            path_models, device,
            resize_w=image_width,
            resize_h=image_height
        )

    elif segmentor_name == 'fast_sam':
        from libs.segmentor import fast_sam_module

        segmentor = fast_sam_module.FastSamClass(
            {'width': image_width, 'height': image_height,
             'mask_height': image_height, 'mask_width': image_width,
             'conf': 0.5, 'model': 'FastSAM-s.pt',
             'imgsz': int(max(image_height, image_width, 480))},
            device=device, traversable_categories=traversable_class_names
        )  # imgsz < 480 gives poorer results

    elif segmentor_name == 'sam2':
        from libs.segmentor import sam2_seg
        assert path_models is not None, f'{path_models=} must be provided for {segmentor_name=}!'
        segmentor = sam2_seg.Seg_SAM2(
            model_checkpoint=path_models, resize_w=image_width, resize_h=image_height)

    elif 'sam21' in segmentor_name:
        sam_kwargs = {}
        if 'pps' in segmentor_name:
            sam_kwargs = {"points_per_side": int(
                segmentor_name.split("_")[-1][3:])}
        from libs.segmentor import sam21
        segmentor = sam21.Seg_SAM21(
            resize_w=image_width, resize_h=image_height, sam_kwargs=sam_kwargs)

    elif segmentor_name == 'sim':
        raise ValueError(
            'Simulator segments not supported in topological mode...')

    else:
        raise NotImplementedError(f'{segmentor_name=} not implemented...')

    return segmentor
