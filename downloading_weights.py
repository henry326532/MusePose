import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_file(session, url, path):
    local_filename = url.split('/')[-1]
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        with open(os.path.join(path, local_filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def download_files(urls, paths):
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(download_file, session, url, path) for url, path in zip(urls, paths)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading files"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Download failed: {e}")

base_dir = 'pretrained_weights'

urls = ['https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        'https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth',
        'https://huggingface.co/TMElyralab/MusePose/resolve/main/MusePose/denoising_unet.pth',
        'https://huggingface.co/TMElyralab/MusePose/resolve/main/MusePose/motion_module.pth',
        'https://huggingface.co/TMElyralab/MusePose/resolve/main/MusePose/pose_guider.pth',
        'https://huggingface.co/TMElyralab/MusePose/resolve/main/MusePose/reference_unet.pth',
        'https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/unet/diffusion_pytorch_model.bin',
        'https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin',
        'https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin'
        ]

paths = ['dwpose', 'dwpose', 'MusePose', 'MusePose', 'MusePose', 'MusePose', 'sd-image-variations-diffusers/unet', 'image_encoder', 'sd-vae-ft-mse']
# Create directories
for path in set(paths):
    os.makedirs(path, exist_ok=True)

# Download files
download_files(urls, paths)
