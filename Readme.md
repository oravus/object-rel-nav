## [ObjectReact [CoRL'25]](https://object-react.github.io/) | [TANGO [ICRA'25]](https://podgorki.github.io/TANGO/) | [RoboHop [ICRA'24]](https://oravus.github.io/RoboHop/)
Code for testing different controllers / trajectory planners for **Object-Relative Navigation**.

## Setup
#### Environment

<details>
  <summary> Setup Conda/Mamba environment with torch, habitat etc. </summary>

```
conda create -n nav
conda activate nav

conda install python=3.9 mamba -c conda-forge
mamba install pip numpy matplotlib pytorch torchvision pytorch-cuda=11.8 opencv=4.6 cmake=3.14.0 habitat-sim withbullet numba=0.57 pyyaml ipykernel networkx h5py natsort open-clip-torch transformers einops scikit-learn kornia pgmpy python-igraph pyvis -c pytorch -c nvidia -c aihabitat -c conda-forge

mamba install -c conda-forge ultralytics

mamba install -c conda-forge tyro faiss-gpu scikit-image ipykernel spatialmath-python gdown utm seaborn wandb kaggle yacs

# install habitat-lab
cd libs/
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab/
git checkout v0.2.4
pip install -e habitat-lab
```
</details>

#### Data
<details>
<summary>
Download official Habitat data and our benchmark trajectories.
</summary>

In `./data/`, sym link the following downloads as subdirs: `hm3d v0.2`, `instance_imagenav_hm3d_v3`, and `hm3d_iin_val`.
- Download official `hm3d v0.2` following instructions [here](https://github.com/matterport/habitat-matterport-3dresearch).
- Download official `InstanceImageNav` challenge dataset from [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip) (Direct Link | ~512 mb)
- Download our test trajectory data `hm3d_iin_val` from [here](https://drive.google.com/file/d/18yhsuz52QvWQ8gQHeWXLAaqoa6T6jk0O/view?usp=sharing). 

</details>

#### Models
<details> 
<summary> Download controller models.
 </summary>

In `model_weights/`:

- ObjectReact: Download pretrained model from [here](https://drive.google.com/file/d/1L0PUetzZrTrjnLFQbU4G0qT2XpxqRQ0Z/view?usp=sharing) [14 MB].
- TANGO: Download depth anything model from [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth) [1.3 GB] and its base vit from [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) [1.3 GB].
- PixNav: Download the original authors' provided checkpoint from our hosting [here](https://drive.google.com/file/d/1QcnwulbuGEsZX_4qmsH9jD4_iWfUNXeX/view?usp=sharing) [208 MB].

</details>

## Experiment
### Navigation
For quickly running a navigation episode using robohop controller (it uses `configs/defaults.yaml` which has comments explaining the parameters):

```
python main.py
```


To use ObjectReact or TANGO controller (`tango.yaml`, or your own config), run as:

```
python main.py -c configs/object_react.yaml
```

Check the output dir `./out/` for `output.log` and visualizations. The above config uses ground truth perception for a quick start, set `goal_source='topological'` and `edge_weight_str='e3d_max'` to use inferred perception.


### Creating RoboHop's Topological Graph
To create a topological graph given a folder of RGB images, please see/run this example script:

```
python scripts/create_maps_hm3d.py ./data/hm3d_iin_val/ fast_sam None 0 1
```


### Cite
ObjectReact:
```
@inproceedings{garg2025objectreact,
  title={ObjectReact: Learning Object-Relative Control for Visual Navigation},
  author={Garg, Sourav and Craggs, Dustin and Bhat, Vineeth and Mares, Lachlan and Podgorski, Stefan and Krishna, Madhava and Dayoub, Feras and Reid, Ian},
  booktitle={Conference on Robot Learning},
  year={2025},
  organization={PMLR}
}
```

TANGO:
```
@inproceedings{podgorski2025tango,
  title={TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals},
  author={Podgorski, Stefan and Garg, Sourav and Hosseinzadeh, Mehdi and Mares, Lachlan and Dayoub, Feras and Reid, Ian},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={2399--2406},
  year={2025},
  organization={IEEE}
}
```

RoboHop:
```
@inproceedings{garg2024robohop,
  title={Robohop: Segment-based topological map representation for open-world visual navigation},
  author={Garg, Sourav and Rana, Krishan and Hosseinzadeh, Mehdi and Mares, Lachlan and S{\"u}nderhauf, Niko and Dayoub, Feras and Reid, Ian},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4090--4097},
  year={2024},
  organization={IEEE}
}
```