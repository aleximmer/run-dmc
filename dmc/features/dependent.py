import pandas as pd
import numpy as np


def group_return_probability(group: pd.Series) -> np.float64:
    """Given the returnQuantities in a group as a Series, divide the number of rows with returns
    (quantity > 0) by the number of rows in total.

    Parameters
    ----------
    group: pd.Series
        Series being the returnQuantities of a group.

    Returns
    -------
    np.float64
        The probability of row that falls in this group to result in a return.

    Example
    -------
    >>> df.groupby('customerID').returnQuantity.apply(group_return_probability)

    """
    return group.astype(bool).sum() / len(group)


def customer_return_probability(df: pd.DataFrame):
    """Calculate likelihood of a specific customer to return a product.

    Parameters
    ----------
    df : pd.DataFrame
    Table containing 'customerID' and 'returnQuantity' columns

    """
    customer_ret_probs = df.groupby('customerID')['returnQuantity'].apply(group_return_probability)
    df['customerReturnProb'] = customer_ret_probs.reindex(df['customerID']).values


def color_return_probability(df: pd.DataFrame):
    """Calculate likelihood of an order with a specific color to result in a return.

    Parameters
    ----------
    df : pd.DataFrame
    Table containing 'customerID' and 'returnQuantity' columns

    """
    color_ret_probs = df.groupby('colorCode')['returnQuantity'].apply(group_return_probability)
    df['colorReturnProb'] = color_ret_probs.reindex(df['colorCode']).values


def size_return_probability(df: pd.DataFrame):
    """Calculate likelihood of an order with a specific sizeCode to result in a return.

    Parameters
    ----------
    df : pd.DataFrame
    Table containing 'customerID' and 'returnQuantity' columns

    """
    color_ret_probs = df.groupby('sizeCode')['returnQuantity'].apply(group_return_probability)
    df['sizeReturnProb'] = color_ret_probs.reindex(df['sizeCode']).values


def product_group_return_probability(df: pd.DataFrame):
    """Calculate likelihood of an order with a specific productGroup to result in a return.

    Parameters
    ----------
    df : pd.DataFrame
    Table containing 'customerID' and 'returnQuantity' columns

    """
    color_ret_probs = df.groupby('productGroup')['returnQuantity'].apply(group_return_probability)
    df['productGroupReturnProb'] = color_ret_probs.reindex(df['productGroup']).values


def binned_color_code(df: pd.DataFrame, deviations=1.0):
    """Bin colorCode column.

    This is to deal with unknown colorCodes (CC's) in the target set by binning the CC range.
    The binning considers outlier CC's by keeping them separate, 1-sized bins.
    Outliers are CC's whose return probability is over one standard deviation away from the mean.
    For our training data (mean: 0.548, std: 0.114) colorCode c is an is an outlier if

        retProb(c) < 0.434 || 0.662 < retProb(c), given deviations = 1.

    Parameters
    ----------
    df : pd.DataFrame
        Table containing 'colorCodes' column to be binned
    deviations : float
        Number of standard deviations a return probability has to differ from the mean to be
        considered an outlier.

    """
    color_code_min = 0
    color_code_max = 9999

    cc_ret_probs = df.groupby('colorCode')['returnQuantity'].apply(group_return_probability)

    mean = cc_ret_probs.mean()
    diff = cc_ret_probs.std() * deviations

    mean_distances = cc_ret_probs.sub(mean).abs()

    bins = [color_code_min]

    # iterate over colorCodes and respective mean distances to collect bins
    for cc, mean_distance in mean_distances.items():
        if mean_distance > diff:
            # add the colorCode as 1-sized bin (current cc and cc + 1)
            # if the last colorCode was added, don't add the current cc, only cc + 1.
            if bins[-1] != cc:
                bins.append(cc)
            bins.append(cc + 1)
    bins.append(color_code_max + 1)

    cut = list(pd.cut(df.colorCode, bins, right=False))
    df['binnedColorCode'] = pd.Series(cut, index=df.index)
