import os
import random
import shutil
from math import ceil, floor
from typing import Tuple, List

import numpy as np
from tqdm import tqdm

from datapack import DataPack


class Shuffle(object):

    def __init__(
            self,
            split_indice: Tuple[float, float, float] = (0.8, 0.1, 0.7),
            task_indice: Tuple[int, int] = (5, 8),
            temporal_indice: Tuple[int, int] = (0.5, 3.0)
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
        self.temporal_indice = temporal_indice

    @staticmethod
    def _relabel_person_id(datapack: DataPack):
        id_lut = list(range(datapack.current_person + 1))
        random.shuffle(id_lut)
        _pack = {}
        for cam_id, protos in datapack.pack.items():
            _pack[cam_id] = {}
            for person_id, imgs in protos.items():
                _pack[cam_id][id_lut[person_id]] = imgs
        datapack.pack = _pack

    @staticmethod
    def _adjust_camera_count(datapack: DataPack, camera_num: int):

        while len(datapack.pack) != camera_num:

            # if the number of camera datapack is more than that of edge node,
            # then merge the last 2 camera datapack into 1 camera datapack.
            if len(datapack.pack) > camera_num:
                sorted_pack = sorted(
                    [(cam, len(person)) for cam, person in datapack.pack.items()],
                    key=lambda t: t[1]
                )
                min_cam_id = sorted_pack[0][0]
                min_cam_persons = datapack.pack[min_cam_id].keys()
                most_diff_cam_id = sorted_pack[1][0]
                most_diff_cam_persons = datapack.pack[most_diff_cam_id].keys()
                for cam_id, _ in sorted_pack:
                    curr_cam_persons = datapack.pack[cam_id].keys()
                    if len(min_cam_persons - curr_cam_persons) > len(min_cam_persons - most_diff_cam_persons):
                        most_diff_cam_id = cam_id
                        most_diff_cam_persons = curr_cam_persons

                # merge two camera id
                for person_id, img_list in datapack.pack[min_cam_id].items():
                    if person_id not in datapack.pack[most_diff_cam_id].keys():
                        datapack.pack[most_diff_cam_id][person_id] = []
                    datapack.pack[most_diff_cam_id][person_id].extend(img_list)

                # remove one camera datapack.
                del datapack.pack[min_cam_id]

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

        # rebase the camera id
        _datapack = {}
        for idx, (camera_id, person_seq) in enumerate(datapack.pack.items(), 0):
            _datapack[idx] = dict(sorted(person_seq.items(), key=lambda ite: ite[0]))
        datapack.pack = dict(sorted(_datapack.items(), key=lambda ite: ite[0]))
        datapack.current_camera = len(_datapack.keys()) - 1

    @staticmethod
    def _sample_person_seq(
            datapack: DataPack,
            task_cnt: int,
            temporal_ratio: float = 0.5,
            temporal_distance: float = 3.0,
    ):
        for cam_id, person_seq in datapack.pack.items():
            task_size = floor(len(person_seq) / task_cnt)
            person_ids = np.array(list(person_seq.keys()))

            # random replace two person id from different tasks
            for swap_cnt in range(int(temporal_ratio * task_size)):
                x = y = 0
                while not 1.0 * task_size <= x - y <= temporal_distance * task_size:
                    x = np.random.randint(0, len(person_seq))
                    y = np.random.randint(0, len(person_seq))
                person_ids[x], person_ids[y] = person_ids[y], person_ids[x]

            # random replace two different tasks
            task_resample_idx = np.arange(task_cnt)
            for swap_cnt in range(int(temporal_ratio * task_cnt)):
                x = y = 0
                while not 1.0 <= x - y <= temporal_distance:
                    x = np.random.randint(0, task_cnt)
                    y = np.random.randint(0, task_cnt)
                task_resample_idx[x], task_resample_idx[y] = task_resample_idx[y], task_resample_idx[x]

            _person_ids = np.zeros_like(person_ids)
            for task_id, source_pop_idx in enumerate(range(0, len(person_seq), task_size)):
                if task_id < task_cnt:
                    target_pop_idx = task_resample_idx[task_id] * task_size
                    _person_ids[source_pop_idx:source_pop_idx + task_size] = \
                        person_ids[target_pop_idx:target_pop_idx + task_size]
                else:
                    _person_ids[source_pop_idx:] = person_ids[source_pop_idx:]
            person_ids = _person_ids

            # sort person ids of each task
            for pop_idx in range(0, len(person_seq), task_size):
                person_ids[pop_idx:pop_idx + task_size] = sorted(person_ids[pop_idx:pop_idx + task_size])

            # plot histogram of person ids distribution
            # from matplotlib import pyplot as plt
            # plt.figure(figsize=(25, 3), dpi=300)
            # for pop_idx in range(0, len(person_seq), task_size):
            #     if pop_idx // task_size < task_cnt:
            #         datas = [person_ids[idx] for idx in range(pop_idx, pop_idx + task_size)]
            #         plt.subplot(1, task_cnt, 1 + pop_idx // task_size)
            #         plt.hist(datas, bins=20, rwidth=5, range=(min(person_ids), max(person_ids)))
            # plt.show()

            # apply the changes in datapack
            _person_seq = {person_id: person_seq[person_id] for person_id in person_ids}
            datapack.pack[cam_id] = _person_seq

    @staticmethod
    def save_imgs(img_path_list: List, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for img_path in img_path_list:
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            shutil.copyfile(img_path, save_path)

    def shuffle_and_save(self, datapack: DataPack, output: str, seed: int = 123):
        np.random.seed(seed)

        # relabel person ids
        self._relabel_person_id(datapack)

        # adjust camera view count if the number of view is less than edge node
        if datapack.current_camera + 1 != self.task_indice[0]:
            self._adjust_camera_count(datapack, self.task_indice[0])

        # shuffle and save the dataset
        self._sample_person_seq(datapack, self.task_indice[1], self.temporal_indice[0], self.temporal_indice[1])
        for cam_id, person_seq in tqdm(datapack.pack.items(), desc="Saving"):
            batch_id = -1
            batch_size = ceil(len(person_seq) / self.task_indice[1])

            empty_query = False
            # divide equally by person_id with each camera
            for idx, (person_id, img_list) in enumerate(person_seq.items()):

                if idx % batch_size == 0 and batch_id + 1 < self.task_indice[1]:
                    if empty_query:
                        print("empty query for: camera {}, batch {}.".format(cam_id, batch_id))
                    empty_query = True
                    batch_id += 1

                task_save_dir = os.path.join(output, f'task-{cam_id}-{batch_id}')
                tr_save_dir = os.path.join(task_save_dir, 'train', f'{person_id}')
                query_save_dir = os.path.join(task_save_dir, 'query', f'{person_id}')
                gallery_save_dir = os.path.join(task_save_dir, 'gallery', f'{person_id}')

                img_list_size = len(img_list)
                np.random.shuffle(img_list)

                # img_list is the total images from current person id and camera.
                # and the train, query and gallery list has following roles:
                # for example the split indice is (0.8, 0.1, 0.7)
                # train  : random sample 80% from img_list;
                # query  : random sample 10% from other cameras that is differ
                #          from gallery and train cameras; if other camera not
                #          exists, then random choice 10% from train img_list.
                # gallery: contain the rest of img_list that has never been used
                #          by train and query dataset, and the 70% image in train
                #          dataset.

                tr_img_list = []
                query_img_list = []
                gallery_img_list = []

                split_pivot = self.split_indice[0] * img_list_size

                # choose the images before split pivot as train images
                tr_img_list.extend(img_list[:ceil(split_pivot)])

                # choose the images between pre-split-pivot and new split_pivot
                pre_split_pivot = split_pivot
                split_pivot = (self.split_indice[0] + self.split_indice[1]) * img_list_size
                query_img_list.extend(img_list[floor(pre_split_pivot):ceil(split_pivot)])

                # choose gallery images from different cameras
                for other_cam_id, other_person_seq in datapack.pack.items():
                    if cam_id == other_cam_id:
                        continue
                    if person_id in other_person_seq.keys():
                        np.random.seed(seed)
                        gallery_img_list.extend(np.random.choice(
                            other_person_seq[person_id],
                            size=int(len(other_person_seq[person_id]) * self.split_indice[2]),
                            replace=False
                        ).tolist())

                if len(tr_img_list):
                    self.save_imgs(tr_img_list, tr_save_dir)
                if len(query_img_list):
                    empty_query = False
                    self.save_imgs(query_img_list, query_save_dir)
                if len(gallery_img_list):
                    self.save_imgs(gallery_img_list, gallery_save_dir)
