import pandas as pd
from tqdm import tqdm

TRAIN_CAT = '../input/train_categorical_2.csv.ftr'
TRAIN_NUM = '../input/train_numeric.csv.ftr'
TRAIN_DAT = '../input/train_date.csv.ftr'


def main(path):
    map_cols = {}

    df = pd.read_feather(path, nthreads=8)

    for col in tqdm(df.columns):
        col_str = ','.join(map(str, df[col].values))
        if col_str in map_cols:
            map_cols[col_str].append(col)
        else:
            map_cols[col_str] = [col]
    return list(map_cols.values())


if __name__ == '__main__':
    for path in [TRAIN_CAT, TRAIN_DAT, TRAIN_NUM]:
        cols = main(path)
        with open('tmp.txt', 'a') as f:
            f.write(f'{path}\n')
            f.write(f'{cols}\n')
