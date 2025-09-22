# File: dataset_h5.py (Versione finale antiproiettile)

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm

class DatasetH5(Dataset):
    def __init__(self, path, img_size=640, clip_range=(-1500, 1500)):
        self.img_size = img_size
        self.clip_min, self.clip_max = clip_range
        self.class_id = 0
        
        self.h5_files = sorted(glob.glob(os.path.join(path, '*.h5')))
        self.n = len(self.h5_files)

        self.labels = []
        self.shapes = []
        
        print(f"Caching and validating labels & shapes from {len(self.h5_files)} files...")
        bad_labels_count = 0
        for h5_path in tqdm(self.h5_files, desc=f"Loading metadata from {path}"):
            try:
                with h5py.File(h5_path, 'r') as f:
                    magnetogram_data = f['magnetogram/data']
                    orig_h, orig_w = magnetogram_data.shape
                    self.shapes.append([orig_h, orig_w])

                    harp_group = f['harp/metadata']
                    image_labels = []
                    for harp_id in harp_group:
                        harp_attrs = harp_group[harp_id].attrs
                        
                        required_keys = ['CRPIX1', 'CRPIX2', 'CRSIZE1', 'CRSIZE2']
                        if not all(key in harp_attrs for key in required_keys):
                            continue

                        # +++ INIZIO BLOCCO DI CONTROLLO "ANTIPROIETTILE" +++
                        # Controlliamo che i dati siano validi PRIMA di usarli
                        w_abs = float(harp_attrs['CRSIZE1'])
                        h_abs = float(harp_attrs['CRSIZE2'])

                        # Se la larghezza o l'altezza sono zero o negative, scartiamo la box
                        if w_abs <= 0 or h_abs <= 0:
                            bad_labels_count += 1
                            continue

                        x_center_norm = float(harp_attrs['CRPIX1']) / orig_w
                        y_center_norm = float(harp_attrs['CRPIX2']) / orig_h
                        width_norm = w_abs / orig_w
                        height_norm = h_abs / orig_h

                        # Controlliamo che le coordinate siano tra 0 e 1
                        if not (0.0 < x_center_norm < 1.0 and 0.0 < y_center_norm < 1.0):
                             bad_labels_count += 1
                             continue
                        # +++ FINE BLOCCO DI CONTROLLO +++

                        image_labels.append([self.class_id, x_center_norm, y_center_norm, width_norm, height_norm])
                    
                    self.labels.append(np.array(image_labels, dtype=np.float32) if image_labels else np.empty((0, 5), dtype=np.float32))

            except Exception as e:
                print(f"Errore grave durante la lettura del file {h5_path}: {e}")
                self.labels.append(np.empty((0, 5), dtype=np.float32))
                self.shapes.append([0, 0])

        if bad_labels_count > 0:
            print(f"ATTENZIONE: Trovate e scartate {bad_labels_count} etichette corrotte (es. dimensioni zero/negative o fuori dall'immagine).")

        self.shapes = np.array(self.shapes, dtype=np.float64)

    def __len__(self):
        return self.n


    def __getitem__(self, index):
        h5_path = self.h5_files[index]
        labels_tensor = torch.from_numpy(self.labels[index])
        
        with h5py.File(h5_path, 'r') as f:
            data = f['magnetogram/data'][:]

            # +++ CONTROLLO ANTI-DEMONE +++
            # Se ci sono valori corrotti (nan/inf) nel file, li neutralizziamo
            if np.isnan(data).any() or np.isinf(data).any():
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            # +++ FINE CONTROLLO +++
            
            clipped_data = np.clip(data, self.clip_min, self.clip_max)
            normalized_data = (clipped_data - self.clip_min) / (self.clip_max - self.clip_min)
            resized_image = cv2.resize(normalized_data, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            image_rgb = np.stack([resized_image] * 3, axis=-1)
            
        image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
        return image_tensor, labels_tensor, h5_path, self.shapes[index]