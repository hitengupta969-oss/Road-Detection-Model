"""
Builds road_images.h5py and road_masks.h5py (the two files the notebook loads).

Steps: download the Massachusetts Roads images listed in Images.txt / Targets.txt,
cut each 1500x1500 image into 256x256 tiles, drop useless tiles, save to .h5py.

Run it in Google Colab (see the steps in the chat). Edit the CONFIG block only.
"""
import os, urllib.request
import numpy as np, cv2, h5py
from tqdm import tqdm

# ---------------- CONFIG ----------------
PROJECT_DIR = "/content/gdrive/MyDrive/Road-Project"   # folder in your Drive that holds Images.txt / Targets.txt
RAW_DIR     = "/content/raw"                           # temporary download area (Colab disk, not Drive)
MAX_IMAGES  = 400      # how many of the 1108 images to use. 400 -> ~9,000 tiles (fits free Colab RAM)
TILE        = 256
# ----------------------------------------

os.makedirs(f"{RAW_DIR}/sat", exist_ok=True)
os.makedirs(f"{RAW_DIR}/map", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/Data", exist_ok=True)

def read_urls(name):
    with open(f"{PROJECT_DIR}/{name}") as f:
        return [l.strip() for l in f if l.strip()]

def download(url, folder):
    dest = f"{folder}/{os.path.basename(url)}"
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception as e:
        print("skip (download failed):", os.path.basename(url), e)
        return None

img_urls = read_urls("Images.txt")[:MAX_IMAGES]
msk_urls = read_urls("Targets.txt")[:MAX_IMAGES]

tiles_img, tiles_msk = [], []
for iu, mu in tqdm(list(zip(img_urls, msk_urls)), desc="download + crop"):
    ip, mp = download(iu, f"{RAW_DIR}/sat"), download(mu, f"{RAW_DIR}/map")
    if not ip or not mp:
        continue
    img  = cv2.imread(ip)                       # BGR
    mask = cv2.imread(mp, 0)
    if img is None or mask is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # notebook/plots expect RGB
    for r in range(0, img.shape[0] - TILE + 1, TILE):        # full tiles only (5x5 = 25 per image)
        for c in range(0, img.shape[1] - TILE + 1, TILE):
            ti, tm = img[r:r+TILE, c:c+TILE], mask[r:r+TILE, c:c+TILE]
            tm = np.where(tm > 127, 255, 0).astype("uint8")   # clean binary mask: 0 or 255
            missing = np.all((ti == 255) | (ti == 0), axis=2).mean()
            if missing > 0.25:                  # >25% blank/missing area (as in the README)
                continue
            if (tm > 0).mean() < 0.01:          # <1% road -> almost nothing to learn from
                continue
            tiles_img.append(ti); tiles_msk.append(tm)
    os.remove(ip); os.remove(mp)                # free Colab disk as we go

print("tiles kept:", len(tiles_img))
with h5py.File(f"{PROJECT_DIR}/Data/road_images.h5py", "w") as f:
    f.create_dataset("all_images", data=np.stack(tiles_img), compression="gzip")
with h5py.File(f"{PROJECT_DIR}/Data/road_masks.h5py", "w") as f:
    f.create_dataset("all_masks", data=np.stack(tiles_msk), compression="gzip")
print("saved to", f"{PROJECT_DIR}/Data/")