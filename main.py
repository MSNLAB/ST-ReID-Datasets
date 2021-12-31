import argparse

from datapack import DataPack
from extractor import dataset_name
from shuffle import Shuffle


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='dataset names')
    parser.add_argument('--roots', type=str, nargs='+', required=True, help='dataset root path')
    parser.add_argument('--output', type=str, required=True, help='dataset processed output path')
    parser.add_argument('--split_indice', type=float, nargs='+', required=False, default=[0.8, 0.1, 0.7],
                        help='train, query and gallery split indice')
    parser.add_argument('--task_indice', type=int, nargs='+', required=False, default=[5, 10],
                        help='edge node and task count')
    parser.add_argument('--temporal_indice', type=float, nargs='+', required=False, default=[0.5, 3.0],
                        help='temporal ratio and temporal distance indice')
    parser.add_argument('--random_seed', type=int, required=False, default=0, help='split seed')
    args = vars(parser.parse_args())

    datasets = args['datasets']
    roots = args['roots']
    output = args['output']
    split_indice = args['split_indice']
    task_indice = args['task_indice']
    temporal_indice = args['temporal_indice']
    random_seed = args['random_seed']

    datapack = DataPack()
    for dataset, root in zip(datasets, roots):
        dataset_name[dataset](datapack, root).process()

    shuffle = Shuffle(split_indice, task_indice, temporal_indice)
    shuffle.shuffle_and_save(datapack, output, random_seed)


if __name__ == '__main__':
    main()
