import os
import numpy as np
from natsort import natsorted
import cv2
import h5py
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import pickle
import sys

import logging
logger = logging.getLogger("[Mapper]") # logger level is explicitly set below by LOG_LEVEL (TODO: better option?)

from libs.experiments import model_loader
from libs.matcher import lightglue as matcher_lg
from libs.common import utils
from libs.common import utils_visualize as utils_viz
from libs.logger.level import LOG_LEVEL
logger.setLevel(LOG_LEVEL)


class MapTopological:
    def __init__(self, imgDir, outDir, cfg={}, segmentor=None):
        self.cfg = self.default_config()
        self.cfg.update(cfg)

        self.imgNames = natsorted(os.listdir(f'{imgDir}'))
        self.imgNames = [f'{imgDir}/{imgName}' for imgName in self.imgNames]
        print(f"{len(self.imgNames)} images in the map directory {imgDir}")

        # convenience variables
        self.W = self.cfg["W"]
        self.H = self.cfg["H"]
        segmentor_name = self.cfg['segmentor_name']
        device = self.cfg['device']

        # init other variables
        self.G, self.nodeID_to_imgRegionIdx, self.G4 = None, None, None

        # subsample images
        # optionally copy them (useful when original images are in a different dir)
        si, se, ss = self.cfg["subsample_si"], self.cfg["subsample_ei"], self.cfg["subsample_step"]
        self.imgNames = self.imgNames[si:se:ss]
        if self.cfg["copy_images"]:
            os.makedirs(f"{outDir}/images", exist_ok=True)
            ext = self.imgNames[0].split('.')[-1]
            for i, imgName in enumerate(self.imgNames):
                os.system(f"cp {imgName} {outDir}/images/{i:04d}.{ext}")

        self.textLabels = self.cfg["textLabels"]
        if len(self.textLabels) > 0:
            assert segmentor_name == 'fast_sam', "Text labels only supported for fast_sam"

        self.h5FullPath = f'{outDir}/nodes_{segmentor_name}.h5'
        if len(self.textLabels) > 0:
            self.h5FullPath = f"{self.h5FullPath[:-3]}_filteredByText.h5"

        if segmentor is None:
            self.segmentor = model_loader.get_segmentor(
                segmentor_name, self.W, self.H, device=device, path_models=self.cfg.get("modelPath", None))
        else:
            self.segmentor = segmentor

        self.matcher_name = self.cfg["matcher_name"]
        if self.matcher_name == "lightglue":
            self.matcher = matcher_lg.MatchLightGlue(
                self.W, self.H, cfg=self.cfg)
        elif self.matcher_name == "sam2":
            self.matcher = self.segmentor

        self.graphPath = f'{self.h5FullPath[:-3]}_graphObject_4_{self.matcher_name}.pickle'

    def load_graph(self):
        G4 = None
        if os.path.exists(self.graphPath):
            logger.info(f"Loading precomputed graph from {self.graphPath}")
            G4 = pickle.load(open(self.graphPath, 'rb'))
        else:
            logger.info(
                f"{self.graphPath=} not found, consider running self.create_map_topo()")
        return G4

    def default_config(self):
        return {
            "W": 320,
            "H": 240,
            "segmentor_name": "fast_sam",
            "modelPath": None,
            "device": "cuda",
            "force_recompute_masks": False,
            "force_recompute_graph": False,
            "matcher_name": "lightglue",
            "textLabels": [],
            "match_area": True,
            "subsample_si": 0,
            "subsample_ei": None,
            "subsample_step": 1,
            "copy_images": False,
            "precompute_path_lengths": True,
            "remove_h5": False,
            "edge_weight_str": None,
            "rewrite_graph_with_allPathLengths": False,
        }

    def create_map_topo(self):
        if (not os.path.exists(self.h5FullPath) and not os.path.exists(self.graphPath)) or self.cfg["force_recompute_masks"]:
            self.create_masks_h5()
        else:
            logger.info(
                f"Using precomputed masks from {self.h5FullPath} or {self.graphPath}")

        if not os.path.exists(self.graphPath) or self.cfg["force_recompute_graph"] or self.cfg["force_recompute_masks"]:
            self.G = self.create_graph_intra_img()
            self.nodeID_to_imgRegionIdx = self.get_nodeID_to_imgRegionIdx(
                self.G)
            self.G4 = self.create_graph_inter_img()
            utils.change_edge_attr(self.G4)
            pickle.dump(self.G4, open(self.graphPath, 'wb'))
            if self.cfg["remove_h5"]:
                os.remove(self.h5FullPath)
        else:
            logger.info(f"Using precomputed graph from {self.graphPath}")
            self.G4 = self.load_graph()

        if self.cfg["precompute_path_lengths"]:
            _ = self.get_precomputed_path_lengths(
                weight=self.cfg['edge_weight_str'])

    def get_precomputed_path_lengths(self, weight):
        # check if precomputed path lengths exist, else compute and save
        allPathLengths = self.G4.graph.get('allPathLengths', {})

        # check weight str of allPathLengths dict
        if weight not in allPathLengths:
            logger.warning(
                f"allPathLengths not found in graph, computing now using {weight=}...")
            allPathLengths.update(
                {weight: self.get_all_paths(self.G4, weight=weight)})
            self.G4.graph['allPathLengths'] = allPathLengths
            if self.cfg["rewrite_graph_with_allPathLengths"]:
                logger.warning(
                    f"Rewriting graph with allPathLengths to {self.graphPath}")
                pickle.dump(self.G4, open(self.graphPath, 'wb'))
        return allPathLengths

    def get_nodeID_to_imgRegionIdx(self, G):
        return np.array([G.nodes[node]['map'] for node in G.nodes()])

    def get_img(self, imgIdx):
        img = cv2.imread(self.imgNames[imgIdx])[:, :, ::-1]
        img = cv2.resize(img, (self.W, self.H))
        return img

    def create_masks_h5(self, compressMask=True):
        with h5py.File(self.h5FullPath, "w") as f:
            # store cfg
            f.attrs['cfg'] = str(self.cfg)
            # loop over images and create a h5py group per image
            for i, _ in enumerate(tqdm(self.imgNames)):
                if isinstance(self.imgNames[0], str):
                    imname = self.imgNames[i]
                    im = cv2.imread(imname)[:, :, ::-1]
                else:
                    imname, im = i, self.imgNames[i]
                masks = self.segmentor.segment(im, textLabels=self.textLabels)
                if self.cfg['segmentor_name'] in ['fast_sam', 'sam2']:
                    masks = masks[0]  # fast_sam returns multiple outputs
                if masks is None:
                    # create a dummy mask array with a single mask of all True
                    masks = [{'segmentation': np.ones(
                        (self.H, self.W), dtype=bool), 'area': self.H*self.W}]
                grp = f.create_group(f"{imname}")
                grp.create_group("masks")
                for j, m in enumerate(masks):
                    for k in m.keys():
                        if k == 'segmentation' and compressMask:
                            rle = utils.mask_to_rle_numpy(m[k][None, ...])[0]
                            grp["masks"].create_dataset(
                                f"{j}/{k}/counts", data=rle['counts'])
                            grp["masks"].create_dataset(
                                f"{j}/{k}/size", data=rle['size'])
                        else:
                            grp["masks"].create_dataset(f"{j}/{k}", data=m[k])

    def create_graph_intra_img(self, intraNbrsAll=False):
        G = nx.Graph()
        # add cfg to graph
        G.graph['cfg'] = self.cfg
        rInds = []
        temporalEdges = []
        with h5py.File(self.h5FullPath, "r") as f:
            for i, _ in enumerate(tqdm(self.imgNames)):
                key = imname = self.imgNames[i]
                # print_hdf5_item_info(f[key])
                masks = [f[key+f'/masks/{k}/']
                         for k in natsorted(f[key+'/masks/'].keys())]

                # TODO: remove line masks? (root cause of the following required filtering)
                # masks = [mask for mask in masks if 0 in mask['bbox'][()][-2:]]

                masks_seg = [mask['segmentation'] for mask in masks]
                masks_seg = [utils.rle_to_mask(seg) for seg in masks_seg]
                mask_cords = np.array(
                    [np.array(np.nonzero(seg)).mean(1)[::-1] for seg in masks_seg])

                uniInds = createFilter(mask_cords, i)
                masks, masks_seg, mask_cords = applyFilter(
                    uniInds, [masks, masks_seg, mask_cords])
                mask_cords = np.array(mask_cords)
                numSegs = len(masks)

                nbrs = create_edges_DT(mask_cords)
                numNodesCurr = len(rInds)
                nodes = [(numNodesCurr+j, {**{"map": [i, j]}, **{k: v[()] for k, v in masks[j].items() if k != 'segmentation'}, **{
                          "segmentation": utils.mask_to_rle_numpy(masks_seg[j][None, ...])[0]}}) for j in range(numSegs)]
                if intraNbrsAll:
                    edges = [(numNodesCurr+j, numNodesCurr+k)
                             for j in range(numSegs) for k in range(j+1, numSegs)]
                else:
                    edges = (np.array(nbrs)+numNodesCurr).tolist()
                G.add_nodes_from(nodes)
                G.add_edges_from(edges)
                # create a single temporal edge using the last node of the previous image and the first node of the current image
                if i != 0:
                    temporalEdges.append(
                        (numNodesCurr-1, numNodesCurr, {'sim': 0}))

                rInds += [[i, j] for j in range(numSegs)]
        G.graph['rft_da_env_arr'] = None
        G.graph['rft_lang_env_arr'] = None
        G.graph['temporalEdges'] = np.array(temporalEdges)
        return G

    def create_graph_inter_img(self):
        intraImage_edges, da_edges, temporal_edges = utils.getSplitEdgeLists(
            self.G, flipSim=True)

        da_edges_rob, _ = self.get_robust_DA_edges(win=3)
        print(f"Num robust DA edges: {len(da_edges_rob)}")
        G4 = utils.modify_graph(self.G, self.G.nodes(
            data=True), da_edges_rob + intraImage_edges)

        print(
            f"Num Nodes and eges: {G4.number_of_nodes()}, {G4.number_of_edges()}")
        return G4

    def get_robust_DA_edges(self, topK=None, win=0):
        imgIdx_s, imgIdx_e = self.nodeID_to_imgRegionIdx[0,
                                                         0], self.nodeID_to_imgRegionIdx[-1, 0] + 1
        da_edges = []
        temporal_edges = []
        for i in tqdm(range(imgIdx_s, imgIdx_e)):
            nodeInds_i = np.argwhere(
                self.nodeID_to_imgRegionIdx[:, 0] == i).flatten()
            if len(nodeInds_i) == 0:
                continue
            nodes_i = [self.G.nodes(data=True)[n] for n in nodeInds_i]
            if win == 0:
                endIdx = imgIdx_e
            else:
                endIdx = min(i+1+win, imgIdx_e)

            qryImg = None
            if self.matcher_name == "sam2":
                qryImg = cv2.resize(cv2.imread(self.imgNames[i])[
                                    :, :, ::-1], (self.matcher.resize_w, self.matcher.resize_h))
                nodes_i = [{"segmentation": utils.rle_to_mask(node["segmentation"]),
                            "bbox": node["bbox"]} for node in nodes_i]

            for j in range(i+1, endIdx):
                nodeInds_j = np.argwhere(
                    self.nodeID_to_imgRegionIdx[:, 0] == j).flatten()
                if len(nodeInds_j) == 0:
                    continue
                nodes_j = [self.G.nodes(data=True)[n] for n in nodeInds_j]
                margin_ij = np.zeros((len(nodeInds_i)))

                if self.matcher_name == "sam2":
                    refImg = cv2.resize(cv2.imread(self.imgNames[j])[
                                        :, :, ::-1], (self.matcher.resize_w, self.matcher.resize_h))
                    nodes_j = [{"segmentation": utils.rle_to_mask(node["segmentation"]),
                                "bbox": node["bbox"]} for node in nodes_j]

                    matchPairsList, _ = self.matcher.track_segments_in_sequence(
                        qryImg, [refImg], nodes_i, [nodes_j], query_frame_idx_in_sequence=0)

                    matches_ij = matchPairsList.T[0]
                    matchesBool = matches_ij != -1
                else:
                    # override with local matching
                    results = self.matcher.matchPair_imgWithMask(
                        self.imgNames[i], self.imgNames[j], nodes_i, nodes_j, visualize=False)
                    if results is None:
                        continue
                    matchesBool, matches_ij, singleBestMatch, _, vizImgs = results

                da_edges.append(np.array([nodeInds_i[matchesBool], nodeInds_j[matches_ij][matchesBool], margin_ij[matchesBool]])[
                                :, np.argsort(-margin_ij[matchesBool])[:topK]])
                # if j==i+1:
                # temporal_edges.append([nodeInds_i[singleBestMatch], nodeInds_j[matches_ij][singleBestMatch], margin_ij[singleBestMatch]])
        if len(da_edges) != 0:
            da_edges = np.concatenate(da_edges, axis=1).T
        # temporal_edges = np.array(temporal_edges)
        da_edges = [(int(e[0]), int(e[1]), {
                     'margin': np.exp(-100*e[2]), 'edgeType': 'da'}) for e in da_edges]
        # temporal_edges = [(int(e[0]),int(e[1]),{'margin':np.exp(-100*e[2]), 'edgeType':'temporal'}) for e in temporal_edges]
        return da_edges, temporal_edges

    def get_all_paths(self, G, weight=None, maxVal=1e6):
        if utils.count_edges_with_given_weight(G, weight) == 0:
            raise ValueError(
                f'No edges found for given {weight=}, found {utils.get_edge_weight_types(G)=}')

        pathLengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
        pathLengths = np.array([[pathLengths[src].get(
            tgt, maxVal) for tgt in G.nodes()] for src in G.nodes()]).astype(np.float16)
        return pathLengths

    def visualize_goal_mask(self, imgIdx, display=False, no_pl=False, goalNodeIdx=None):
        img = self.get_img(imgIdx)
        if self.nodeID_to_imgRegionIdx is None:
            self.nodeID_to_imgRegionIdx = self.get_nodeID_to_imgRegionIdx(
                self.G4)
        ni2ir = self.nodeID_to_imgRegionIdx
        nodeInds = np.argwhere(ni2ir[:, 0] == imgIdx).flatten()
        masks = utils.nodes2key(nodeInds, 'segmentation', self.G4)

        if no_pl:
            # visualize masks regardless of path_lengths
            pls = np.arange(len(nodeInds))
        else:
            weight = self.cfg['edge_weight_str']
            allPathLengths = self.get_precomputed_path_lengths(weight=weight)
            allPathLengths = allPathLengths[weight]
            if goalNodeIdx is None:
                # approximate last node as goal node
                goalNodeIdx = len(ni2ir)-1
            pls = np.array([allPathLengths[n, goalNodeIdx] for n in nodeInds])
            pls = utils.normalize_pls(pls)

        colors, norm = utils_viz.value_to_color(pls, cm_name='viridis')
        vizImg = utils_viz.drawMasksWithColors(img, masks, colors)
        if display:
            plt.imshow(vizImg)
            plt.colorbar()
            plt.show()
        return vizImg

    def visualize_graph_node_index(self, nodeIdx):
        if self.nodeID_to_imgRegionIdx is None:
            self.nodeID_to_imgRegionIdx = self.get_nodeID_to_imgRegionIdx(
                self.G4)
        imgIdx, segIdx = self.nodeID_to_imgRegionIdx[nodeIdx]
        img = self.get_img(imgIdx)
        masks = utils.nodes2key([nodeIdx], 'segmentation', self.G4)

        colors, norm = utils_viz.value_to_color(np.arange(1), cm_name='viridis')
        vizImg = utils_viz.drawMasksWithColors(img, masks, colors)
        plt.imshow(vizImg)
        plt.show()


def getNbrsDelaunay(tri, v):
    indptr, indices = tri.vertex_neighbor_vertices
    v_nbrs = indices[indptr[v]:indptr[v+1]]
    v_nbrs = [[v, u] for u in v_nbrs]
    return v_nbrs


def removeDuplicateNbrPairs(nbrList):
    return list(set(tuple(sorted(tup)) for tup in nbrList))


def create_edges_DT(mask_cords):
    if len(mask_cords) > 3:
        tri = Delaunay(mask_cords)
        nbrs, nbrsLists = [], []
        for v in range(len(mask_cords)):
            nbrsList = getNbrsDelaunay(tri, v)
            nbrsLists.append(nbrsList)
            nbrs += nbrsList
        nbrs = removeDuplicateNbrPairs(nbrs)
    else:
        numCords = len(mask_cords)
        nbrs = [[u, v] for u in range(numCords)
                for v in range(u + 1, numCords)]
    return np.array(nbrs)


def createFilter(mask_cords, i):
    # remove non-unique mask_coords, keeping the first one
    _, unique_idx = np.unique(mask_cords, axis=0, return_index=True)
    numVary = abs(len(mask_cords) - len(unique_idx))
    if numVary > 0:
        print(f"Img {i} - Removed {numVary} duplicate mask coords")
    return unique_idx


def applyFilter(idx, objectsList):
    newObjectsList = []
    for obj in objectsList:
        newObjectsList.append([obj[i] for i in idx])
    return newObjectsList


# Example usage, run from main repo:
# python -m libs.mapper.map_topo /path/to/imgDir /path/to/outDir segmentor_name /path/to/sam_model (this can be omitted to use fast_sam)
if __name__ == "__main__":

    imgDir = sys.argv[1]
    outDir = sys.argv[2]
    segmentor_name = sys.argv[3]
    if len(sys.argv) > 4:
        modelPath = sys.argv[4]
    elif segmentor_name in ['sam', 'sam2']:
        assert segmentor_name in [
            'sam', 'sam2'], "modelPath is required for sam and sam2 segmentor_names"
    else:
        modelPath = None

    if not os.path.exists(outDir):
        os.makedirs(outDir, exist_ok=True)

    cfg = {"W": 320, "H": 240, "device": "cuda", "segmentor_name": segmentor_name, "modelPath": modelPath,
           "force_recompute_masks": True, "force_recompute_graph": True, "matcher_name": "lightglue",
           "textLabels": ["floor", "ceiling"],
           #    "textLabels": [],
           "match_area": True,
           "subsample_si": 0, "subsample_ei": None, "subsample_step": 1,
           "copy_images": False}
    mapper = MapTopological(imgDir, outDir=outDir, cfg=cfg)

    mapper.create_map_topo()

    # visualize goal masks for reference images
    for i in range(len(mapper.imgNames)):
        mapper.visualize_goal_mask(i, display=True, no_pl=False)
