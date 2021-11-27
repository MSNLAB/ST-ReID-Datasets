import os
import re
from typing import Dict

from tqdm import tqdm

from datapack import DataPack
from extractor.module import ExtractorModule


class Extractor(ExtractorModule):
    """
    dataset_path should like:
    |-- bounding_box_train
    |   | -- 0002_c1s1_000451_03.jpg
    |   | -- 0002_c1s1_000551_01.jpg
    |   | -- ...
    |-- bounding_box_test
    |   | -- 0000_c1s1_000151_01.jpg
    |   | -- 0000_c1s1_000376_03.jpg
    |   | -- ...
    |-- gt_bbox
    |   | -- 0001_c1s1_001051_00.jpg
    |   | -- 0001_c1s1_002301_00.jpg
    |   | -- ...
    |-- gt_query
    |-- query
    |   | -- 0001_c1s1_001051_00.jpg
    |   | -- 0001_c2s1_000301_00.jpg
    |   | -- ...
    |-- readme.txt
    """

    def __init__(self, datapack: DataPack, root: str, download: bool = False, **kwargs):
        super(Extractor, self).__init__(datapack, root, download, **kwargs)
        self.datapack = datapack
        self.root = root
        self.img_list = []  # [ (camera, person, img_path) ]

    def process(self, **kwargs):
        if not os.path.exists(self.root):
            raise ValueError(f"Market1501 dataset path '{self.root}' could not be found.")

        self._process("bounding_box_train")
        self._process("bounding_box_test")
        self._process("gt_bbox")
        self._process("query")

        # save images in datapack
        camera_register_map = {}
        person_register_map = {}
        for camera, person, img_path in self.img_list:
            if camera not in camera_register_map.keys():
                camera_register_map[camera] = self.datapack.register_camera()
            if person not in person_register_map.keys():
                person_register_map[person] = self.datapack.register_person()
            camera_id = camera_register_map[camera]
            person_id = person_register_map[person]
            self.datapack.add_image_path(person_id, camera_id, img_path)

    def _process(self, base_name: str):
        base_path = os.path.join(self.root, base_name)

        # find all images by person id
        for img_name in tqdm(os.listdir(base_path), desc=f'Market1501 {base_name} search'):
            if re.match(r'(\d{4})_c(\d)s(\d)_(\w+)(\.jpg)', img_name) is not None:
                img_path = os.path.join(base_path, img_name)
                img_info = self._extract_detail(img_name)
                cam_id = img_info['camera']
                person_id = img_info['id']
                if person_id > 0:
                    self.img_list.append((cam_id, person_id, img_path))

    @staticmethod
    def _extract_detail(img_name: str) -> Dict:
        name_details = img_name.split('_', 3)
        return {
            'id': int(name_details[0]),
            'camera': int(name_details[1][1]),
            'sequence': int(name_details[1][3]),
            'frame': name_details[2]
        }
