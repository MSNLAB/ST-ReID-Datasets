from datapack import DataPack


class ExtractorModule(object):

    def __init__(self, datapack: DataPack, root: str, download: bool = False, **kwargs):
        pass

    def process(self, **kwargs):
        raise NotImplementedError
