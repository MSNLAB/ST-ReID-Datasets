import os
import re
from typing import Dict

from tqdm import *

from datapack import DataPack
from extractor.module import ExtractorModule


class Extractor(ExtractorModule):
    """
    dataset_path should like:
    |-- 001_01_1.png
    |-- 001_01_2.png
    |-- 001_01_3.png
    |-- ...
    """

    def __init__(self, datapack: DataPack, root: str, download: bool = False, **kwargs):
        super().__init__(datapack, root, download, **kwargs)
        self.datapack = datapack
        self.root = root
        self.img_list = []  # [ (camera, person, img_path) ]

    def process(self, **kwargs):
        if not os.path.exists(self.root):
            raise ValueError(f"PKU-ReID dataset path '{self.root}' could not be found.")

        for img_name in tqdm(os.listdir(self.root), desc=f'PKU-ReID search'):
            if re.match(r'(\d{3})_(\d{2})_(\d)(\.png)', img_name):
                img_path = os.path.join(self.root, img_name)
                img_info = self._extract_detail(img_name)
                cam_id = img_info['camera']
                person_id = img_info['id']
                self.img_list.append((cam_id, person_id, img_path))

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

    @staticmethod
    def _extract_detail(img_name: str) -> Dict:
        name_details = img_name.split('_', 3)
        return {
            'id': int(name_details[0]),
            'camera': int(name_details[1]),
            'frame': name_details[2]
        }
