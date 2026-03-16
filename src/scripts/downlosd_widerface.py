#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import zipfile
import requests
from tqdm import tqdm

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(local_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(local_path)) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def download_widerface_val(data_dir='../../dataset/widerface', raw_dir='../../dataet/raw'):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    files = [
        {
            'url': 'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip',
            'filename': 'WIDER_val.zip'
        },
        {
            'url': 'https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip',
            'filename': 'wider_face_split.zip'
        }
    ]

    for file in files:
        zip_path = os.path.join(raw_dir, file['filename'])
        if not os.path.exists(zip_path):
            print(f"Downloading {file['filename']} ...")
            try:
                download_file(file['url'], zip_path)
            except Exception as e:
                print(f"Download failed: {e}")
                print("Please manually download from http://shuoyang1213.me/WIDERFACE/")
                return
        else:
            print(f"{file['filename']} already exists, skip.")

        # 解压到 data_dir
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)
        print(f"Extracted {file['filename']} to {data_dir}")

if __name__ == '__main__':
    download_widerface_val()