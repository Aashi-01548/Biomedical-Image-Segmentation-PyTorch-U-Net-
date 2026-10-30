import os
import requests
import torch

os.makedirs("weights", exist_ok=True)

CELL_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Iu482aowoEaP9MgNqhyJ_w/model-epo300.pt"
BRAIN_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/HATpl02lA0ykn9aAU9K6sA/checkpoint-epoch-29.pt"


def download(url, path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)


download(CELL_URL, "weights/unet_cells.pt")
download(BRAIN_URL, "weights/brain_checkpoint.pt")

# The brain file holds extra training info. Keep only the weights.
ckpt = torch.load("weights/brain_checkpoint.pt", map_location="cpu", weights_only=False)
torch.save(ckpt["model_state_dict"], "weights/unet_brain.pt")
os.remove("weights/brain_checkpoint.pt")

print("Done. Files in weights/:", os.listdir("weights"))