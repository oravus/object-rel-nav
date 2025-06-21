import os
from natsort import natsorted
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import time
import sys
from typing import Optional

import logging
logger = logging.getLogger("[Goal Generator]") # logger level is explicitly set below by LOG_LEVEL (TODO: Tidy up!)

from libs.localizer import loc_topo
from libs.planner_global import plan_topo
# from libs.tracker import track_topo
from libs.common import utils, utils_data
from libs.common import utils_visualize as utils_viz
from libs.logger.level import LOG_LEVEL
logger.setLevel(LOG_LEVEL)

class Goal_Gen:

    def __init__(self, W: int, H: int, G, map_path, poses: Optional = None, task_type: str = "original", cfg: dict = {}):
        self.W = W
        self.H = H
        self.max_pl = 100
        self.cfg = cfg
        self.qIter = -1
        self.goalMask_default = self.max_pl * np.ones((self.H, self.W))  # default value for invalid goal segments
        self.G = G
        self.map_path = map_path
        self.imgNames = natsorted(os.listdir(f"{self.map_path}/images/"))
        self.set_variables_init()

        self.positions = None
        if poses is not None:
            self.positions = np.array([pose.position for pose in poses])

        utils.change_edge_attr(self.G)
        self.nodeID_to_imgRegionIdx = np.array([self.G.nodes[node]['map'] for node in self.G.nodes()])

        self.goalNodeIdx = self.cfg.get("goalNodeIdx", None)
        if self.goalNodeIdx == -1:
            self.goalNodeIdx = self.G.number_of_nodes() - 1
        elif self.goalNodeIdx is None:
            self.goalNodeIdx = utils_data.get_goalNodeIdx(self.map_path, self.G, self.nodeID_to_imgRegionIdx, task_type)
            if self.goalNodeIdx is None:
                raise ValueError
        self.goalImgIdx = self.nodeID_to_imgRegionIdx[self.goalNodeIdx][0] # TODO: find max if list of goal nodes
        self.cfg.update({"goalImgIdx": self.goalImgIdx})

        self.localizer = loc_topo.LocalizeTopological(
            f"{self.map_path}/images", self.G, self.W, self.H, mapImgPositions=self.positions, cfg=self.cfg
        )

        self.planner_g = plan_topo.PlanTopological(self.G, self.goalNodeIdx, self.cfg)

        self.do_track = self.cfg["do_track"]
        if self.do_track:
            self.tracker = None

    def set_variables_init(self):
        self.qryCoords, self.qryMasks, self.pls, self.pls_min, self.local_pls_min, self.pls_median, self.local_pls_median, self.matchPairs, self.coords = [None for _ in range(9)]
        self.goalMask = self.goalMask_default.copy()

    def set_variables_default(self):
        self.pls = self.max_pl * np.ones(len(self.qryCoords), int)
        self.pls_min = self.pls.copy()
        self.pls_median = self.pls.copy()
        self.local_pls_min = self.pls.copy()
        self.local_pls_median = self.pls.copy()
        self.coords = self.qryCoords.copy()
        self.matchPairs = np.column_stack([np.arange(len(self.qryCoords)), self.pls])
        self.goalMask = self.goalMask_default.copy()

    def get_goal_mask(self, qryImg, qryNodes, qryPosition: Optional = None, remove_mask=None, refImgInds=None):
        # resume localization for the new observation
        self.localizer.lost = False
        self.set_variables_init()
        self.qIter += 1
        logger.info(f"Iter: {self.qIter}")
        printTime = False
        t1 = time.time()
        # self.qryNodes = self.segmentor.segment(qryImg)
        self.qryNodes = qryNodes

        if printTime: print(f"Segmentation time: {time.time() - t1:.2f}s")

        if self.qryNodes is None:
            self.localizer.lost = True
            return self.goalMask
        else:
            self.qryMasks = utils.nodes2key(self.qryNodes, 'segmentation')
            if remove_mask is not None:
                self.qryMasks *= remove_mask[None, ...]
            self.qryCoords = utils.nodes2key(self.qryNodes, 'coords')
            if printTime: print(f"Seg Proc time: {time.time() - t1:.2f}s")

            # LOCALIZE
            self.matchPairs = self.localizer.localize(qryImg, self.qryNodes, qryPosition, refImgInds=refImgInds)
            if printTime: print(f"Localization time: {time.time() - t1:.2f}s")

            # TRACK
            if self.do_track:
                curr_to_matched_pls_history = self.tracker.track(qryImg, self.qryNodes)
            if printTime: print(f"Tracking time: {time.time() - t1:.2f}s")

        if self.localizer.lost:
            self.set_variables_default()
            return self.goalMask

        # PLAN
        self.pls, nodesClose2Goal = self.planner_g.get_pathLengths_matchedNodes(self.matchPairs[:, 1])

        # minimize over repeated qry nodes (efficient impl, might be hard to grasp)
        qryIdx = np.arange(len(self.qryCoords))
        matchesMask = self.matchPairs[:, 0][:, None] == qryIdx[None, :] # N_qry_repeat x N_qry_original
        pls_min_inds = (matchesMask * (self.pls.max()-self.pls[:, None])).argmax(0) # N_qry_original
        self.pls_min = self.pls[pls_min_inds]
        pls_min_ref_node_inds = self.matchPairs[pls_min_inds, 1]

        # set pl to maxval for unmatched nodes
        unmatched_node_inds = matchesMask.sum(0) == 0
        if sum(unmatched_node_inds) > 0:
            self.pls_min[unmatched_node_inds] = self.max_pl

        self.coords = self.qryCoords[self.matchPairs[:, 0]]

        if printTime: print(f"Planning time: {time.time() - t1:.2f}s")

        # TRACK UPDATES PLAN
        if self.do_track:
            self.pls_median, self.local_pls_median = self.tracker.update(self.pls_min, qryImg, self.qryNodes, self.localizer.qry_features)
        else:
            self.pls_median = self.pls_min.copy()
            

        for qIdx in qryIdx:
            self.goalMask[self.qryMasks[qIdx]] = self.pls_median[qIdx]

        if printTime: print(f"Tracked Plan & Goal mask time: {time.time() - t1:.2f}s")

        return self.goalMask

    def visualize_goal_mask(self, qryImg, display: bool = False):
        pls = utils.normalize_pls(self.pls)
        colors, norm = utils_viz.value_to_color(pls, cm_name='viridis')
        vizImg = utils_viz.drawMasksWithColors(qryImg, self.qryMasks[self.matchPairs[:, 0]], colors)
        if display:
            plt.imshow(vizImg)
            plt.colorbar()
            plt.show()
        return vizImg

    def loadImg(self, idx):
        img = cv2.imread(f"{self.map_path}/images/{self.imgNames[idx]}")[:, :, ::-1]
        img = cv2.resize(img, (self.W, self.H))
        return img

    def visualize_goal_node(self, goalNodeIdx=None, display=False):
        if goalNodeIdx is None:
            goalNodeIdx = self.goalNodeIdx
        imgIdx, regIdx = self.nodeID_to_imgRegionIdx[goalNodeIdx]
        mask = utils.rle_to_mask(self.G.nodes[goalNodeIdx]['segmentation'])
        img = self.loadImg(imgIdx)
        img_masked = img.copy()
        img_masked[mask] = [255, 0, 0]
        if display:
            plt.imshow(img)
            plt.imshow(img_masked, alpha=0.5)
            plt.show()
        return img_masked
        

# Example usage
# python -m libs.goal_generator.goal_gen /path/to/map/ <segmentor_type=sam|sam2|fast_sam> /path/to/model/ (optional for fast_sam)

if __name__ == "__main__":

    mapPath = sys.argv[1]
    graphFileName = sys.argv[2]
    segmentor_type = sys.argv[3]
    if len(sys.argv) > 4:
        modelPath = sys.argv[4]
    else:
        modelPath = None

    W, H = 320, 240
    if segmentor_type == 'sam':
        from libs.segmentor import sam
        segmentor = sam.Seg_SAM(modelPath, "cuda", resize_w=W, resize_h=H)
    elif segmentor_type == 'fast_sam':
        from libs.segmentor import fast_sam_module
        segmentor = fast_sam_module.FastSamClass({'width':W, 'height':H, 'mask_height':H, 'mask_width':W, 'conf':0.25, 'model':'FastSAM-s.pt', 'imgsz':max(H,W,480)}) # imgsz < 480 gives poorer results
    elif segmentor_type == 'sam2':
        from libs.segmentor import sam2_seg
        segmentor = sam2_seg.Seg_SAM2(model_checkpoint=modelPath, resize_w=W, resize_h=H)
    else:
        raise ValueError(f"Segmentor {segmentor_type} not recognized")

    matcher_name = "lightglue" #"sam2"
    cfg = {"matcher_name": matcher_name}
    if cfg["matcher_name"] == "sam2":
        cfg["sam2_tracker"] = segmentor
        # sam2 for seg+match during localization and sam2+LG during mapping

    graphPath = f"{mapPath}/{graphFileName}"
    if not os.path.exists(graphPath):
        assert f"{graphPath} does not exist, to create it, uncomment this assertion!"
        logger.warning(f"Graph for segmentor {segmentor_type} and matcher {matcher_name} not found. Creating one now!")
        from libs.mapper import map_topo
        cfg_map = {}
        cfg_map.update(cfg)
        cfg_map['match_area'] = True
        cfg_map['textLabels'] = []#["floor", "ceiling"]
        mapper = map_topo.MapTopological(f"{mapPath}/images/", outDir=mapPath, cfg=cfg_map, segmentor=segmentor_type)
        mapper.create_map_topo()

    G = pickle.load(open(graphPath, 'rb'))

    cfg.update({"goalNodeIdx": len(G.nodes()) - 1})
    goalie = Goal_Gen(W, H, G, mapPath, None, cfg=cfg)

    # loop over images
    imgList = natsorted(os.listdir(f"{mapPath}/images/"))
    for i, imgName in enumerate(imgList):
        # if i < 7:
            # continue
        # print(i)
        qryImg = cv2.imread(f"{mapPath}/images/{imgName}")[:, :, ::-1]
        qryImg = cv2.resize(qryImg, (goalie.W, goalie.H))

        qryNodes, _, textSim = segmentor.segment(qryImg, textLabels = ["floor"])
        # print(f"Text similarity: {textSim}")
        goalMask = goalie.get_goal_mask(qryImg, qryNodes)
        _ = goalie.visualize_goal_mask(qryImg, display=True)
