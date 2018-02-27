import pandas as pd
from ast import literal_eval
import numpy as np
import os
from typing import List


def format_X_y(df: pd.DataFrame, X_col: str='X', y_col: str='y'):
    """
    Converts input and output data into a format usable by SKLearn.
    :param df: DataFrame containing input and output data.
    :param X_col: Name of the column which holds input data.
    :param y_col: Name of the column which holds outputs data.
    :return: Input and output numpy arrays for use with SKLearn regression models.
    """
    X = time_series_to_X(df=df, X_col=X_col)
    y = np.array(df[y_col])

    return X, y


def time_series_to_X(df: pd.DataFrame, X_col: str='X'):
    """
    Converts input data into a format usable with SKLearn.
    :param df: DataFrame containing the X values to be processed.
    :param X_col: Name of the column in which the input data is located.
    :return: numpy.array of X values.
    """
    if isinstance(df['X'], str):
        df['X'] = df['X'].apply(lambda x: literal_eval(x))
        return np.asarray(df['X'].tolist())
    elif isinstance(df['X'], list):
        return np.asarray(df['X'])
    else:
        raise ValueError("Unsupported X type found at df['X']")


def split_train_test(df: pd.DataFrame, percent: float=0.8):
    """
    Creates a train and test set by random sampling where 'percent' of the initial
    data is used for training.
    :param df: The DataFrame to split.
    :param percent: The percentage of data to use for training.
    :return: A DataFrame consisting of train data, and a DataFrame consisting of test/validation data.
    """
    df = df.sample(frac=1).reset_index(drop=True)
    num_rows = len(df)

    split_index = int(percent * num_rows)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index + 1:]

    return train_df, test_df


def split_time_series(df: pd.DataFrame, chunk_size: int, target_col: float='price'):
    """
    Splits time-series data in chunks of chunk_size, where the last index is y, and the rest
    of the data is X.
    :param df: DataFrame containing the time-series data.
    :param chunk_size: Size of arrays to be created for training/testing.
    :param target_col: The column name of the data to be used.
    :return: DataFrame consisting of two columns: X and y.
    """
    df_size = len(df)
    rows = []
    for i in range(df_size - chunk_size - 1):
        row = {'X': df.iloc[i:i+chunk_size-1][target_col].values.tolist(), 'y': df.iloc[i+chunk_size][target_col]}
        rows.append(row)
    return pd.DataFrame(rows)


def combine_data_from_dir(save_name: str, search_dir: str, save_dir: str, chunk_sizes: List[int]=[10]):

    master_df = pd.DataFrame()

    for chunk_size in chunk_sizes:
        for filename in os.listdir(search_dir):
            if filename.endswith(".csv"):
                print(f"Formatting {filename}")

                df = pd.read_csv(f"{search_dir}/{filename}", encoding='utf-8')

                df = split_time_series(df, chunk_size=10)

                master_df = master_df.append(df)

        print(f"Saving df to {save_dir}/{save_name}_chunk_{chunk_size}.csv")
        master_df.to_csv(f"{save_dir}/{save_name}_chunk_{chunk_size}.csv", index=False, encoding='utf-8')

def format_percent_change(df):

    rows = []

    df['X'] = df['X'].apply(lambda x: literal_eval(x))

    for i, row in df.iterrows():
        price = row['y']
        prev_price = row['X'][-1]
        change = abs(price - prev_price)
        change_percent = change / price
        row['pct'] = change_percent

        changes = []

        for x in range(len(row['X']) - 1):
            curr = row['X'][x]
            next = row['X'][x+1]
            change = abs(curr - next)
            changes.append(change)

        row['changes'] = changes

        rows.append(row)

    new_df = pd.DataFrame(rows)

    new_df = new_df[['changes', 'pct']].copy()
    new_df['X'] = new_df['changes']
    new_df['y'] = new_df['pct']
    new_df = new_df[['X', 'y']].copy()

    return new_df

def main():
    main_dir = "C:/Users/God/data/stock_data/"
    load_dir = "C:/Users/God/data/stock_data/day_stocks/formatted/training/day_stock_train_list_chunk_10.csv"
    save_dir = "C:/Users/God/data/stock_data/day_stocks/formatted/training/day_stock_train_list_percent_chunk_10.csv"
    day_stock_dir = os.path.join(main_dir, "day_stocks/testing")

    # combine_data_from_dir("day_stock_test_list", search_dir=day_stock_dir, save_dir=save_dir)

    df = pd.read_csv(load_dir)
    df = format_percent_change(df)
    df.to_csv(save_dir, index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
