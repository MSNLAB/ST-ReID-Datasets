import os
import re

from tqdm import *

from datapack import DataPack
from extractor.module import ExtractorModule


class Extractor(ExtractorModule):
    """
    dataset_path should like:
    |-- multi_shot
    |    |--  cam_a
    |    |    |--  person_0001
    |    |    |--  person_0002
    |    |    |--  ...
    |    |--  cam_b
    |    |    |--  person_0001
    |    |    |--  person_0002
    |    |    |--  ...
    |-- single_shot
    |    |--  cam_a
    |    |--  cam_b
    |-- readme.txt
    """

    def __init__(self, datapack: DataPack, root: str, download: bool = False, **kwargs):
        super(Extractor, self).__init__(datapack, root, download, **kwargs)
        self.datapack = datapack
        self.root = root
        self.img_list = []  # [ (camera, person, img_path) ]

    def process(self, **kwargs):
        if not os.path.exists(self.root):
            raise ValueError(f"PRID2011 dataset path '{self.root}' could not be found.")

        self._process_multi_shot("cam_a")
        self._process_multi_shot("cam_b")
        self._process_single_shot("cam_a")
        self._process_single_shot("cam_b")

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

    def _process_multi_shot(self, cam_name: str):
        cam_path = os.path.join(self.root, 'multi_shot', cam_name)
        # find all images by person id
        for id_name in tqdm(os.listdir(cam_path), desc=f'PRID2011 multi-shot {cam_name} search'):
            id_path = os.path.join(cam_path, id_name)
            for img_name in os.listdir(id_path):
                if re.match(r'(\d{4}).png', img_name):
                    img_path = os.path.join(id_path, img_name)
                    self.img_list.append((cam_name, id_name, img_path))

    def _process_single_shot(self, cam_name: str):
        cam_path = os.path.join(self.root, 'single_shot', cam_name)
        # find all images by person id
        for img_name in tqdm(os.listdir(cam_path), desc=f'PRID2011 single-shot {cam_name} search'):
            if re.match(r'person_(\d{4}).png', img_name):
                id_name = os.path.splitext(img_name)[0]
                img_path = os.path.join(cam_path, img_name)
                self.img_list.append((cam_name, id_name, img_path))
