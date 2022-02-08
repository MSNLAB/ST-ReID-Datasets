import argparse
import os
import shutil
from typing import List

from datapack import DataPack
from extractor import dataset_name
from shuffle import Shuffle


def save_imgs(img_path_list: List, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img_path in img_path_list:
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        shutil.copyfile(img_path, save_path)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='dataset names')
    parser.add_argument('--roots', type=str, nargs='+', required=True, help='dataset root path')
    parser.add_argument('--output', type=str, required=True, help='dataset processed output path')
    parser.add_argument('--split_indice', type=float, nargs='+', required=False, default=[0.6, 0.4, 0.7],
                        help='train, query and gallery split indice')
    parser.add_argument('--task_indice', type=int, required=False, default=8, help='edge node and task count')
    parser.add_argument('--random_seed', type=int, required=False, default=0, help='split seed')
    args = vars(parser.parse_args())

    datasets = args['datasets']
    roots = args['roots']
    output = args['output']
    task_indice = args['task_indice']
    split_indice = args['split_indice']
    random_seed = args['random_seed']

    for dataset, root in zip(datasets, roots):
        datapack = DataPack()
        dataset_name[dataset](datapack, root).process()
        shuffle = Shuffle(split_indice, task_indice)
        shuffle.shuffle_and_save(dataset, datapack, output, random_seed)


if __name__ == '__main__':
    main()
