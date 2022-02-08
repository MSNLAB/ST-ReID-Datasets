import math
import os
import shutil
from typing import Tuple, List

import numpy as np

from datapack import DataPack


class Shuffle(object):

    def __init__(
            self,
            split_indice: Tuple[float, float, float] = (0.8, 0.1, 0.7),
            task_indice: Tuple[int, int] = (5, 8)
    ):
        self.split_indice = split_indice
        self.task_indice = task_indice

    @staticmethod
    def save_imgs(img_path_list: List, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for img_path in img_path_list:
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            shutil.copyfile(img_path, save_path)

    def shuffle_and_save(self, dataset, datapack: DataPack, output: str, seed: int = 123):
        np.random.seed(seed)
        batch_size = math.ceil(len(datapack.pack.keys()) / self.task_indice)

        for idx in range(0, len(datapack.pack.keys()), batch_size):
            person_ids = list(datapack.pack.keys())[idx:idx + batch_size]
            for person_id in person_ids:
                img_list = []
                for cid, pic_list in datapack.pack[person_id].items():
                    img_list.extend(pic_list)
                np.random.shuffle(img_list)

                tr_img_start = 0
                tr_img_end = tr_img_start + math.floor(len(img_list) * self.split_indice[0])

                query_img_start = math.ceil(len(img_list) * self.split_indice[0])
                query_img_end = query_img_start + math.floor(math.ceil(len(img_list) * self.split_indice[1]))

                tr_imgs = img_list[tr_img_start:tr_img_end]
                query_imgs = img_list[query_img_start:query_img_end]
                gallery_imgs = img_list[query_img_end:] + tr_imgs[:math.ceil(len(img_list) * self.split_indice[2])]

                if len(tr_imgs):
                    self.save_imgs(tr_imgs, f'{output}/{dataset}-{idx // batch_size}/train/{person_id}')
                if len(query_imgs):
                    self.save_imgs(query_imgs, f'{output}/{dataset}-{idx // batch_size}/query/{person_id}')
                if len(gallery_imgs):
                    self.save_imgs(gallery_imgs, f'{output}/{dataset}-{idx // batch_size}/gallery/{person_id}')
