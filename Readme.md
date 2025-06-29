## Setup | Conda/Mamba
```
conda create -n nav
conda activate nav

conda install python=3.9 mamba -c conda-forge
mamba install numpy matplotlib pip pytorch torchvision pytorch-cuda=11.8 opencv=4.6 cmake=3.14.0 habitat-sim withbullet numba=0.57 pyyaml ipykernel networkx h5py natsort open-clip-torch transformers einops scikit-learn kornia pgmpy python-igraph pyvis -c pytorch -c nvidia -c aihabitat -c conda-forge

mamba install -c conda-forge ultralytics

[optional] mamba install -c conda-forge tyro faiss-gpu scikit-image ipykernel spatialmath-python gdown utm seaborn wandb kaggle yacs

# install habitat-lab
cd libs/
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab/
git checkout v0.2.4
pip install -e habitat-lab


```

## Data
In `./data/`, sym link the following downloads as subdirs: `hm3d v0.2`, `instance_imagenav_hm3d_v3`, and `hm3d_iin_val`.
- Download official `hm3d v0.2` following instructions [here](https://github.com/matterport/habitat-matterport-3dresearch).
- Download official `InstanceImageNav` challenge dataset from [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip) (Direct Link | ~512 mb)
- Download our test trajectory data `hm3d_iin_val` from [here](https://drive.google.com/file/d/18yhsuz52QvWQ8gQHeWXLAaqoa6T6jk0O/view?usp=sharing). 

#### Models
Download depth anything model weights from [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth) and its base vit from [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) and place them in `model_weights/`

## Experiment
### Navigation
For quickly running a navigation episode using robohop controller (it uses `configs/defaults.yaml` which has comments explaining the parameters):

```
python main.py
```


To use TANGO controller (or your own config), run:

```
python main.py -c configs/tango.yaml
```

Check the output dir `./out/` for `output.log` and visualizations.

### Creating RoboHop's Topological Graph
To create a topological graph given a folder of RGB images, please see/run this example script:

```
python scripts/create_maps_hm3d.py ./data/hm3d_iin_val/ fast_sam None 0 1
```


