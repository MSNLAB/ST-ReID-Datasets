from typing import List, Union


class DataPack(object):
    def __init__(self) -> None:
        super().__init__()
        self.current_person = -1
        self.current_camera = -1
        self.img_cnt = 0
        self.pack = {}  # { Person: { Camera: [Image] } }

    def register_camera(self) -> int:
        self.current_camera += 1
        return self.current_camera

    def register_person(self) -> int:
        self.current_person += 1
        self.pack[self.current_person] = {}
        return self.current_person

    def add_image_path(self, person_id: int, camera_id: int, image_paths: Union[List[str], str]):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        self.img_cnt += len(image_paths)
        if person_id not in self.pack.keys():
            self.pack[person_id][camera_id] = []
        if camera_id not in self.pack[person_id].keys():
            self.pack[person_id][camera_id] = []
        self.pack[person_id][camera_id].extend(image_paths)
