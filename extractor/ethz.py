import os
import re

from tqdm import *

from datapack import DataPack
from extractor.module import ExtractorModule


class Extractor(ExtractorModule):
    """
    dataset_path should like:
    |-- seq1
    |    |--  p001
    |    |--  p002
    |    |--  ...
    |-- seq2
    |    |--  p001
    |    |--  p002
    |    |--  ...
    |-- seq3
    |    |--  p001
    |    |--  p002
    |    |--  ...
    |-- Readme.txt
    """

    def __init__(self, datapack: DataPack, root: str, download: bool = False, **kwargs):
        super().__init__(datapack, root, download, **kwargs)
        self.datapack = datapack
        self.root = root
        self.img_list = []  # [ (camera, person, img_path) ]

    def process(self, **kwargs):
        if not os.path.exists(self.root):
            raise ValueError(f"ETHZ dataset path '{self.root}' could not be found.")

        self._process('seq1')
        self._process('seq2')
        self._process('seq3')

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

    def _process(self, seq_name: str):
        seq_path = os.path.join(self.root, seq_name)

        # find all images by person id
        for id_name in tqdm(os.listdir(seq_path), desc=f'ETHZ {seq_name} search'):
            id_path = os.path.join(seq_path, id_name)
            for img_name in os.listdir(id_path):
                if re.match(r'frame(\d{4})Person(\d{2}).png', img_name):
                    img_path = os.path.join(id_path, img_name)
                    self.img_list.append((0, seq_name + id_name, img_path))
