import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


focal_length_x = 1298.07361
focal_length_y = 1282.95703
cx = 593.435798
cy = 395.75851

INPUT_DIR = './images'
OUTPUT_DIR = './output'
DATASET = 'kitti'  # Let's not pick a fight with the model's dataloader


def process_images(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.png')) + glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    for image_path in tqdm(image_paths, desc="Processing Images"):
        #try:
        color_image = Image.open(image_path).convert('RGB')
        color_image = color_image.resize((1024, 768), Image.LANCZOS)
        original_width, original_height = color_image.size
        image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        # print(image_tensor.size(), torch.max(image_tensor), torch.min(image_tensor))
        with torch.inference_mode():
            pred = model(image_tensor)

            if isinstance(pred, dict):

                pred = pred.get('metric_depth', pred.get('out'))
            elif isinstance(pred, (list, tuple)):

                pred = pred[-1]

            pred = pred.squeeze().detach().cpu().numpy()

        # Resize depth to final size
        resized_pred = Image.fromarray(pred).resize((original_width, original_height), Image.NEAREST)
        # resized_pred.show()

        x, y = np.meshgrid(np.arange(original_width), np.arange(original_height))
        x = (x - cx) / focal_length_x
        y = (y - cy) / focal_length_y
        z = np.array(resized_pred)
        print(z.shape, x.shape)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0
        print(np.max(points[:, 0]), np.min(points[:, 0]), np.max(points[:, 1]), np.min(points[:, 1]), np.max(points[:, 2]), np.min(points[:, 2]))
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        #  pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".ply"), pcd)

        #except Exception as e:
         #   print(f"Error processing {image_path}: {e}")


def main(model_name, pretrained_resource):
    config = get_config(model_name, "infer", None)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    process_images(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_outdoor.pt', help="Pretrained resource to use for fetching weights.")

    args = parser.parse_args()
    main(args.model, args.pretrained_resource)
