import os
import shutil
from math import ceil
from typing import Tuple, List

import numpy as np
from tqdm import tqdm

from datapack import DataPack


class Shuffle(object):

    def __init__(
            self,
            split_indice: Tuple[float, float, float] = (0.8, 0.1, 0.7),
            task_indice: Tuple[int, int] = (5, 10),
    ):
        """
        Shuffle the Datapack images to fcl tasks
        :param datapack: datapack with different people and cameras
        :param split_indice: tuple with 3 elements as following
            (0) train dataset split indice;
            (1) query dataset split indice;
            (2) gallery dataset split indice.
            [note] if the sum of train and query is less than 1.0, then
            the rest part of data will be given to gallery dataset.
        :param task_indice: tuple with 2 elements as following
            (0) client count;
            (1) task count each client have.
        """
        self.split_indice = split_indice
        self.task_indice = task_indice

    @staticmethod
    def _adjust_camera_count(datapack: DataPack, camera_num: int):

        while len(datapack.pack) != camera_num:

            # if the number of camera datapack is more than that of edge node,
            # then merge the last 2 camera datapack into 1 camera datapack.
            if len(datapack.pack) > camera_num:
                # sorted_pack = sorted(
                #     [(cam, len(person)) for cam, person in datapack.pack.items()],
                #     key=lambda t: t[1]
                # )
                # first_last_camera_id = sorted_pack[-1][0]
                # second_last_camera_id = sorted_pack[-2][0]
                first_last_camera_id = list(datapack.pack.keys())[-1]
                second_last_camera_id = list(datapack.pack.keys())[-2]

                # copy all person ids from last camera datapack to
                # the second last camera pack.
                for person_id, img_list in datapack.pack[first_last_camera_id].items():
                    if person_id not in datapack.pack[second_last_camera_id].keys():
                        datapack.pack[second_last_camera_id][person_id] = []
                    datapack.pack[second_last_camera_id][person_id].extend(img_list)

                # remove the last camera datapack.
                del datapack.pack[first_last_camera_id]
                datapack.current_camera -= 1

            # if the number of camera datapack is less than that of edge node,
            # then split the biggest camera datapack into 2 camera datapack.
            else:
                max_cam_id, max_len = max(
                    [(cam, len(person)) for cam, person in datapack.pack.items()],
                    key=lambda t: t[1]
                )
                new_cam_id = datapack.register_camera()

                # copy the top half person ids from biggest camera datapack to
                # the new camera pack.
                trans_num = 0
                for person_id, img_list in datapack.pack[max_cam_id].items():
                    datapack.pack[new_cam_id][person_id] = img_list
                    if (trans_num := trans_num + 1) >= max_len / 2:
                        break

                # remove the transferred person ids from old camera datapack.
                for person_id, img_list in datapack.pack[new_cam_id].items():
                    del datapack.pack[max_cam_id][person_id]

    @staticmethod
    def save_imgs(img_path_list: List, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for img_path in img_path_list:
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            shutil.copyfile(img_path, save_path)

    def shuffle_and_save(self, datapack: DataPack, output: str, seed: int = 123):
        # adjust camera view count if the number of view is less than edge node
        if datapack.current_camera + 1 != self.task_indice[0]:
            self._adjust_camera_count(datapack, self.task_indice[0])

        # shuffle and save the dataset
        for cam_id, person_seq in tqdm(datapack.pack.items(), desc="Saving"):
            batch_id = -1
            batch_size = ceil(len(person_seq) / self.task_indice[1])

            # divide equally by person_id with each camera
            for idx, (person_id, img_list) in enumerate(person_seq.items()):
                if idx % batch_size == 0:
                    batch_id += 1
                task_save_dir = os.path.join(output, f'task-{cam_id}-{batch_id}')
                tr_save_dir = os.path.join(task_save_dir, 'train', f'{person_id}')
                query_save_dir = os.path.join(task_save_dir, 'query', f'{person_id}')
                gallery_save_dir = os.path.join(task_save_dir, 'gallery', f'{person_id}')

                img_list_size = len(img_list)
                np.random.seed(seed)
                np.random.shuffle(img_list)

                # train, query and gallery image list has following roles:
                # for example the split indice is (0.8, 0.1, 0.7)
                # train  : random sample 80% from image list;
                # query  : random sample 10% from image list that
                #          is differ from train datasets;
                # gallery: contain the rest of image list that never been used
                #          by train and query dataset, and the 70% image in
                #          train dataset.
                tr_split_pivot = self.split_indice[0] * img_list_size
                query_split_pivot = (self.split_indice[0] + self.split_indice[1]) * img_list_size

                tr_img_list = img_list[:int(tr_split_pivot)]
                query_img_list = img_list[int(tr_split_pivot): int(query_split_pivot)]
                gallery_img_list = np.random.choice(
                    tr_img_list,
                    size=int(img_list_size * self.split_indice[2]),
                    replace=False
                ).tolist() + img_list[int(query_split_pivot):]

                self.save_imgs(tr_img_list, tr_save_dir)
                self.save_imgs(query_img_list, query_save_dir)
                self.save_imgs(gallery_img_list, gallery_save_dir)
