# RUN: python scripts/create_maps_hm3d.py \
#       </path/to/hm3d_iin_val/> \
#       <segmentor_name=sam|sam2|fast_sam> \
#       </path/to/segment-anything/> (set to None for fast_sam)
#       <startIdx> <endIdx> (for epidodes in the range [startIdx, endIdx))
import os
from natsort import natsorted
import sys
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from libs.logger import default_logger
default_logger.setup_logging()
from libs.mapper.map_topo import MapTopological
from libs.experiments import model_loader

episodesDir = sys.argv[1]
segmentor_name = sys.argv[2]
modelPath = sys.argv[3]
startIdx = sys.argv[4]
endIdx = sys.argv[5]

episodeNames = natsorted(os.listdir(episodesDir))


cfg = {
    "W": 320, "H": 240, "device": "cuda", "segmentor_name": segmentor_name, "modelPath": modelPath,
    "force_recompute_masks": True, 'match_area': True,
    'matcher_name': 'lightglue',
        # storage efficiency
        "remove_h5": True, "precompute_path_lengths": False, 'edge_weight_str': None,
    # "textLabels": ["ceiling", "floor"],
}
segmentor = model_loader.get_segmentor(cfg["segmentor_name"], cfg["W"], cfg["H"], cfg["device"], path_models=cfg["modelPath"])

for ei, episode in enumerate(episodeNames[int(startIdx):int(endIdx)]):
    imgDir = f"{episodesDir}/{episode}/images/"
    outDir = f"{episodesDir}/{episode}/"

    if cfg.get("textLabels") is None:
        textLableStr = ""
    else:
        textLableStr = "_filteredByText"
    if os.path.exists(f"{outDir}/nodes_{segmentor_name}{textLableStr}_graphObject_4_{cfg['matcher_name']}.pickle"):
        print("Exists: ", ei + int(startIdx), episode)
        continue
    else:
        print("Processing: ", ei + int(startIdx), episode)
    mapper = MapTopological(imgDir, outDir=outDir, cfg=cfg, segmentor=segmentor)

    try:
        mapper.create_map_topo()
    except Exception as e:
        print(f"Error: {e}")
        continue
