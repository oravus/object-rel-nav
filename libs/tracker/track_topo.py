
import numpy as np

import logging
logger = logging.getLogger("[Tracker]") # logger level is explicitly set below by LOG_LEVEL

from libs.matcher import lightglue as matcher_lg
from libs.common import utils

from libs.logger.level import LOG_LEVEL
logger.setLevel(LOG_LEVEL)


class TrackTopological:
    def __init__(self, W, H, max_pl, cfg):
        self.max_history = 8
        self.img_history = []
        self.nodes_history = []
        self.ft_history = []
        self.global_pls_history = [] # stores global path length (pl) values
        self.local_pls_history = [] # stores local pl values (i.e., intra-image normalized pl)
        self.max_pl = max_pl

        self.curr_to_matched_global_pls_history = None
        self.curr_to_matched_local_pls_history = None

        self.matcher_name = cfg.get("matcher_name", "lightglue")
        if self.matcher_name == "lightglue":
            self.matcher = matcher_lg.MatchLightGlue(W, H, cfg=cfg)

    def track(self, currImg, currNodes):
        if len(self.img_history) == 0:
            return None

        matchPairsList, _, _ = self.matcher.matchPair_imgWithMask_multi(
            currImg, self.img_history, currNodes, self.nodes_history, ftTgtList=self.ft_history)

        matched_global_pls_history = np.concatenate(
            [self.global_pls_history[i][matchPairs[:, 1]] for i, matchPairs in enumerate(matchPairsList)])
        matched_local_pls_history = np.concatenate(
            [self.local_pls_history[i][matchPairs[:, 1]] for i, matchPairs in enumerate(matchPairsList)])

        self.curr_to_matched_global_pls_history = np.column_stack(
            [np.vstack(matchPairsList)[:, 0], matched_global_pls_history])
        self.curr_to_matched_local_pls_history = np.column_stack(
            [np.vstack(matchPairsList)[:, 0], matched_local_pls_history])

        return self.curr_to_matched_global_pls_history

    def compute_updated_pls(self, curr_pls, history):
        if history is None:
            return curr_pls

        qryIdx = np.arange(len(curr_pls))
        history = np.vstack([history, np.column_stack([qryIdx, curr_pls])])
        qInds_h, pls_h = history[:, 0], history[:, 1]

        pls_median = self.max_pl * np.ones(len(qryIdx))
        for qIdx in qryIdx:
            pls_h_qIdx = pls_h[np.argwhere(qInds_h == qIdx).flatten()]
            inliers = pls_h_qIdx < self.max_pl
            if inliers.sum() > 0:
                pls_median[qIdx] = np.median(pls_h_qIdx[inliers])
            else:
                pls_median[qIdx] = self.max_pl
        return pls_median

    def update(self, curr_global_pls, currImg, currNodes, currFt):

        global_pls = self.compute_updated_pls(curr_global_pls, self.curr_to_matched_global_pls_history)

        # TODO: fix scaling and tracking
        local_pls = 99 * utils.normalize_pls_new(global_pls, scale_factor=1, outlier_value=self.max_pl)
        local_pls = self.compute_updated_pls(local_pls, self.curr_to_matched_local_pls_history)

        self.global_pls_history.append(global_pls)
        self.local_pls_history.append(local_pls)
        self.img_history.append(currImg)
        self.nodes_history.append(currNodes)
        for key in currFt.keys():
            currFt[key] = currFt[key].detach().cpu().numpy()
        self.ft_history.append(currFt)

        if len(self.img_history) > self.max_history:
            self.img_history.pop(0)
            self.nodes_history.pop(0)
            self.ft_history.pop(0)
            self.global_pls_history.pop(0)
            self.local_pls_history.pop(0)
        
        return global_pls, local_pls