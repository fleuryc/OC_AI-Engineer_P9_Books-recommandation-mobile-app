"""Helper functions, not project specific."""

import json
from typing import Union

import pandas as pd


def drop_impossible_values(
    dataframe: pd.DataFrame,
    constraints=dict[str, dict[str, Union[int, float]]],
) -> pd.DataFrame:
    """
    Drop values from a dataframe that have impossible or unlikely values.

    :param dataframe: The dataframe to be filtered.
    :param constraints: A dictionary of constraints to be applied.

    :return: The filtered dataframe.

    Example:
    constraints = {
        'age': {
            'min': 18,
            'max': 60
        }
    }
    """
    for col in dataframe.columns:
        if col in constraints:
            dataframe = dataframe[
                dataframe[col].between(constraints[col]["min"], constraints[col]["max"])
            ]
    return dataframe


def one_hot_encode_list_variables(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    One-hot encode list variables.

    https://cmpoi.medium.com/a-quick-tutorial-to-encode-list-variables-125ba4040325

    - for each list variable
        - decode JSON values to list
        - make a dataframe of one-hot encoded values
        - append to original dataframe

    Args:
        df (pd.DataFrame): dataframe to encode
        columns (list[str]): list of columns to encode

    Raises:
        Exception: columns values should be (JSON encoded) lists of strings

    Returns:
        pd.DataFrame:  dataframe with encoded columns
    """
    df = df.copy()
    for col in columns:
        if not isinstance(df[col][0], list):
            df[col] = df[col].replace("[]", "null")  # replace empty list with null
            df[col] = df[col].apply(json.loads)  # convert string to list

        if not isinstance(df[col][0], list):
            raise Exception(f"{col} is not a list")

        categories_df = (
            pd.get_dummies(
                pd.DataFrame(
                    [
                        x
                        if x is not None
                        else ["__EMPTY__"]  # replace None with empty list
                        for x in df[col].tolist()
                    ]
                ).stack(),
                prefix=col,
            )
            .groupby(level=0)
            .sum()
        )

        categories_df.drop(
            columns=[col for col in categories_df.columns if col.endswith("__EMPTY__")],
            errors="ignore",
            inplace=True,
        )  # remove empty list

        df = pd.concat([df, categories_df], axis=1)

    return df
