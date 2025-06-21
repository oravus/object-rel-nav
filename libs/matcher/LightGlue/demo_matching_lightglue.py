import h5py, os, cv2
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
from pathlib import Path
import time

def multiclass_vis(class_labels, img_to_viz, num_of_labels, np_used=False, alpha=0.5):
    _overlay = img_to_viz.astype(float) / 255.0
    if np_used:
        viz = cmap(class_labels / num_of_labels)[..., :3]
    else:
        class_labels = class_labels.detach().cpu().numpy().astype(float)
        viz = cmap((class_labels / num_of_labels))[..., :3]
    _overlay = alpha * viz + (1 - alpha) * _overlay
    s_overlay = cv2.cvtColor(np.float32(_overlay), cv2.COLOR_BGR2RGB)

    return _overlay


def get_vis_anns(anns, img_to_viz, nbrs=None):
    count = 1
    dum = anns[0]['segmentation']
    img = np.zeros((dum.shape[0], dum.shape[1]))
    for ann in anns:
        m = ann['segmentation']
        img[m] = count
        count += 1
    _overlay = multiclass_vis(img, img_to_viz, count, np_used=True)
    if nbrs is not None:  # ignore this for now
        _overlay = draw_segment_lines(_overlay, anns, nbrs)
    return _overlay


def showSegFromH5(h5filename, ims, imgIdx, regIdx=None, cfg=None, doPlot=True, dataDir="./"):
    with h5py.File(h5filename, 'r') as f:
        if isinstance(ims[0], str):
            imname = ims[imgIdx]
            im = cv2.imread(f'{dataDir}/{imname}')
            print(im.shape)
        else:
            imname, im = imgIdx, ims[imgIdx]
        key = f"{imname}"
        if regIdx is None:
            masks = [f[key + f'/masks/{k}/'] for k in natsorted(f[key + '/masks/'].keys())]
        else:
            masks = [f[key + f'/masks/{regIdx}/']]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (cfg['desired_width'], cfg['desired_height']))
        im = get_vis_anns(masks, im.copy())
        if doPlot:
            plt.imshow(im)
            plt.show()
    return im


def print_hdf5_item_info(item, indent=""):
    if isinstance(item, h5py.Group):
        print(f"{indent}Group: {item.name}")
        for subitem_name in item:
            subitem = item[subitem_name]
            print_hdf5_item_info(subitem, indent + "  ")
    elif isinstance(item, h5py.Dataset):
        print(f"{indent}Dataset: {item.name}")
        print(f"{indent}  Datatype: {item.dtype}")
        print(f"{indent}  Shape: {item.shape}")
    else:
        print(f"{indent}Unknown item: {item.name}")


def match_all_blobs(image_0, all_masks_0, image_1, all_masks_1, device):
    _, im_height, im_width = image_0.shape
    num_masks_0, num_masks_1 = len(all_masks_0), len(all_masks_1)
    matched_kp_numbers = torch.zeros((num_masks_0, num_masks_1), dtype=int, device=device)
    matched_kp_scores = torch.zeros((num_masks_0, num_masks_1), dtype=float, device=device)

    for i, mask0 in enumerate(all_masks_0):
        mask0_seg = torch.tensor(np.array(mask0['segmentation']).reshape(1, im_height, im_width).repeat(3, axis=0),
                                 device=device)
        masked_image_0 = image_0.to(device) * mask0_seg.to(device)
        feats0 = extractor.extract(masked_image_0)
        for j, mask1 in enumerate(all_masks_1):
            mask1_seg = torch.tensor(np.array(mask1['segmentation']).reshape(1, im_height, im_width).repeat(3, axis=0),
                                     device=device)
            masked_image_1 = image_1.to(device) * mask1_seg.to(device)
            feats1 = extractor.extract(masked_image_1)

            matches01 = matcher({'image0': feats0, 'image1': feats1})
            matches01 = rbd(matches01)  # remove batch dimension
            matches = matches01['matches']
            scores = matches01['scores']
            matched_kp_numbers[i][j] = matches.shape[0]
            if matches.shape[0] == 0:
                matched_kp_scores[i][j] = 0
            else:
                matched_kp_scores[i][j] = torch.mean(scores)


    return matched_kp_numbers, matched_kp_scores


def match_query_blob(image_0, query_mask, image_1, all_masks_1, device):
    _, im_height, im_width = image_0.shape
    num_masks_1 = len(all_masks_1)
    matched_kp_numbers = torch.zeros((1, num_masks_1), dtype=int, device=device)
    matched_kp_scores = torch.zeros((1, num_masks_1), dtype=float, device=device)

    query_mask_seg = torch.tensor(np.array(query_mask['segmentation']).reshape(1, im_height, im_width).repeat(3, axis=0),
                             device=device)
    masked_image_0 = image_0.to(device) * query_mask_seg.to(device)
    feats0 = extractor.extract(masked_image_0)
    for j, mask1 in enumerate(all_masks_1):
        mask1_seg = torch.tensor(np.array(mask1['segmentation']).reshape(1, im_height, im_width).repeat(3, axis=0),
                                 device=device)
        masked_image_1 = image_1.to(device) * mask1_seg.to(device)
        feats1 = extractor.extract(masked_image_1)

        matches01 = matcher({'image0': feats0, 'image1': feats1})
        matches01 = rbd(matches01)  # remove batch dimension
        matches = matches01['matches']
        scores = matches01['scores']
        matched_kp_numbers[0][j] = matches.shape[0]
        if matches.shape[0] == 0:
            matched_kp_scores[0][j] = 0
        else:
            matched_kp_scores[0][j] = torch.mean(scores)

    return matched_kp_numbers, matched_kp_scores


if __name__ == '__main__':

    torch.set_grad_enabled(False)
    images = Path('data/images/')

    cmap = cm.get_cmap("jet")

    with h5py.File("./data/GP_dl_env_nodes.h5", 'r') as f:
        print("Image names as keys", f.keys())
        print("\ndata for one image:")
        print_hdf5_item_info(f['Image000.jpg'])

    dataDir = "./data/images/"
    h5File = "./data/GP_dl_env_nodes.h5"
    ims = sorted(os.listdir(dataDir))
    cfg = {"desired_width": 320, "desired_height": 240}
    # _ = showSegFromH5(h5File,ims,imgIdx=0,regIdx=None,cfg=cfg,dataDir=dataDir)
    # _ = showSegFromH5(h5File,ims,imgIdx=1,regIdx=None,cfg=cfg,dataDir=dataDir)

    feats = []
    masks_all_imgs = []
    regIdx = None
    f = h5py.File(h5File, 'r')
    for imgIdx, imgName in enumerate(ims):
        print(imgIdx, imgName)
        im = cv2.imread(f'{dataDir}/{imgName}')
        key = f"{imgName}"
        feats.append(f[key + f'/rft_clip'])

        if regIdx is None:
            masks = [f[key + f'/masks/{k}/'] for k in natsorted(f[key + '/masks/'].keys())]
        else:
            masks = [f[key + f'/masks/{regIdx}/']]
        masks_all_imgs.append(masks)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (cfg['desired_width'], cfg['desired_height']))
        im = get_vis_anns(masks, im.copy())
        if False:
            plt.imshow(im)
            plt.show()

    # Load extractor and matcher module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # use SuperPoint features
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    image0 = load_image(images / 'Image000.jpg', resize=(240, 320)).to(device)
    image1 = load_image(images / 'Image001.jpg', resize=(240, 320)).to(device)

    masks0 = masks_all_imgs[0]
    masks1 = masks_all_imgs[1]

    # matching a query mask from image0 against all the masks from image1
    start_time = time.time()
    matched_kp_numbers, matched_kp_scores = match_query_blob(image0, masks0[0], image1, masks1, device)
    print(f"--- Matching a query blob Matrix Estimation: {(time.time() - start_time)} seconds ---")
    print(matched_kp_numbers, matched_kp_scores)

    # # matching all the masks from image0 against all the masks from image1
    # start_time = time.time()
    # matched_kp_numbers, matched_kp_scores = match_all_blobs(image0, masks0, image1, masks1, device)
    # print(f"--- Matching all blobs Matrix Estimation: {(time.time() - start_time)} seconds ---")
    # print(matched_kp_numbers, matched_kp_scores)
