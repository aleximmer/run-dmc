import pandas as pd
import numpy as np


def customer_return_probability(df: pd.DataFrame):
    """Add a column that contains the likelihood of the customer to return a product.
    Calculate this column by grouping for 'customerID' and dividing the number of rows with
    returns by the number of total rows.

    Parameters
    ----------
    df : pd.DataFrame
        Table containing 'customerID' and 'returnQuantity' columns

    """
    def group_ret_prob(group):
        return group.astype(bool).sum() / len(group)

    customer_ret_probs = df.groupby('customerID').returnQuantity.apply(group_ret_prob)
    customer_ret_probs = customer_ret_probs.reindex(df.customerID)
    df['customerReturnProb'] = customer_ret_probs.values


def color_return_probability(df: pd.DataFrame):
    returned_articles = df.groupby(['colorCode']).returnQuantity.sum()
    bought_articles = df.groupby(['colorCode']).quantity.sum()
    color_return_prob = returned_articles / bought_articles
    df['colorReturnProb'] = pd.Series(list(color_return_prob.loc[df.colorCode]), index=df.index)


def size_return_probability(df: pd.DataFrame):
    returned_articles = df.groupby(['sizeCode']).returnQuantity.sum()
    bought_articles = df.groupby(['sizeCode']).quantity.sum()
    size_return_prob = returned_articles / bought_articles
    df['sizeReturnProb'] = pd.Series(list(size_return_prob.loc[df.sizeCode]), index=df.index)


def product_group_return_probability(df: pd.DataFrame):
    returned_articles = df.groupby(['productGroup']).returnQuantity.sum()
    bought_articles = df.groupby(['productGroup']).quantity.sum()
    product_group_return_prob = returned_articles / bought_articles
    df['productGroupReturnProb'] = pd.Series(list(product_group_return_prob.loc[df.productGroup]), index=df.index)


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

    def ret_prob(d: pd.Series) -> np.float64:
        return d.astype(bool).sum() / len(d)

    cc_ret_probs = df.groupby('colorCode')['returnQuantity'].apply(ret_prob)

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
