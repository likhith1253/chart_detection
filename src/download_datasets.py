import os
import shutil
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def process_plotqa(zip_path, extract_dir, raw_images_dir):
    print("Extracting PlotQA...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Heuristic to find image files
    image_files = list(Path(extract_dir).rglob('*.png'))
    print(f"Found {len(image_files)} PlotQA images. Copying and renaming...")
    
    for i, img_path in enumerate(tqdm(image_files[:500])): # Limit to 500 for demo/speed if needed, or follow instructions
        new_name = f"plotqa_{i+1:05d}.png"
        shutil.copy(img_path, raw_images_dir / new_name)

def process_dvqa(tar_path, extract_dir, raw_images_dir):
    print("Extracting DVQA...")
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_dir)
    
    image_files = list(Path(extract_dir).rglob('*.png'))
    print(f"Found {len(image_files)} DVQA images. Copying and renaming...")
    
    for i, img_path in enumerate(tqdm(image_files[:500])):
        new_name = f"dvqa_{i+1:05d}.png"
        shutil.copy(img_path, raw_images_dir / new_name)

def main():
    temp_dir = Path("data/datasets/staging")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    raw_images_dir = Path('data/raw_images')
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    
    plotqa_url = "https://github.com/NiteshMethani/PlotQA/releases/download/v1.0/plotqa_images.zip"
    dvqa_url = "https://github.com/kushalkafle/DVQA_dataset/releases/download/v1.0/dvqa_images.tar.gz"
    
    plotqa_zip = temp_dir / "plotqa_images.zip"
    dvqa_tar = temp_dir / "dvqa_images.tar.gz"
    
    if not plotqa_zip.exists():
        print("Downloading PlotQA...")
        download_file(plotqa_url, plotqa_zip)
    
    if not dvqa_tar.exists():
        print("Downloading DVQA...")
        download_file(dvqa_url, dvqa_tar)
        
    process_plotqa(plotqa_zip, temp_dir / "plotqa_extracted", raw_images_dir)
    process_dvqa(dvqa_tar, temp_dir / "dvqa_extracted", raw_images_dir)

    # Cleanup
    # shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
