import pandas as pd
import numpy as np
import holidays

COLOR_CODE_MIN = 0
COLOR_CODE_MAX = 9999


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to DataFrame.
    Calls methods that each add a feature in form of a column to the data.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned table training data

    Returns
    -------
    pd.DataFrame
        Feature-enriched table
    """
    df['productPrice'] = df.price / df.quantity
    df['totalSavings'] = df.rrp - df.productPrice
    df['relativeSavings'] = (1 - df.productPrice / df.rrp).fillna(1.)
    df['orderYear'] = df.orderDate.apply(lambda x: x.year)
    df['orderMonth'] = df.orderDate.apply(lambda x: x.month)
    df['orderDay'] = df.orderDate.apply(lambda x: x.day)
    df['orderWeekDay'] = df.orderDate.apply(lambda x: x.dayofweek)
    df['orderDayOfYear'] = df.orderDate.apply(lambda x: x.dayofyear)
    df['orderWeek'] = df.orderDate.apply(lambda x: x.week)
    df['orderWeekOfYear'] = df.orderDate.apply(lambda x: x.weekofyear)
    df['orderQuarter'] = df.orderDate.apply(lambda x: x.quarter)
    df['orderTotalDay'] = df.orderDate.apply(total_day)
    df['orderSeason'] = df.orderDate.apply(date_to_season)
    df['orderIsOnGermanHoliday'] = df.orderDate.apply(lambda x: 1 if x in holidays.DE() else 0)
    df['colorReturnProb'] = color_return_probability(df)
    df['sizeReturnProb'] = size_return_probability(df)
    df['customerReturnProb'] = customer_return_probability(df)
    df['productGroupReturnProb'] = product_group_return_probability(df)
    df['surplusArticleQuantity'] = same_article_surplus(df)
    df['surplusArticleSizeQuantity'] = same_article_same_size_surplus(df)
    df['surplusArticleColorQuantity'] = same_article_same_color_surplus(df)
    df['totalOrderShare'] = total_order_share(df)
    df['voucherSavings'] = voucher_saving(df)
    df['binnedColorCode'] = binned_color_code(df)
    # df['voucherFirstUsedDate'] = pd.to_datetime(df.t_voucher_firstUsedDate_A).apply(total_day)
    # df['voucherLastUsedDate'] = pd.to_datetime(df.t_voucher_lastUsedDate_A).apply(total_day)
    df['customerAvgUnisize'] = df.t_customer_avgUnisize.astype(np.int)
    return df


def remove_features(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.drop('t_voucher_firstUsedDate_A', 1)
    # df = df.drop('t_voucher_lastUsedDate_A', 1)
    df = df.drop('t_customer_avgUnisize', 1)
    return df


def color_return_probability(df: pd.DataFrame) -> pd.Series:
    returned_articles = df.groupby(['colorCode']).returnQuantity.sum()
    bought_articles = df.groupby(['colorCode']).quantity.sum()
    color_return_prob = returned_articles / bought_articles
    return pd.Series(list(color_return_prob.loc[df.colorCode]), index=df.index)


def size_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    returned_articles = df.groupby(['sizeCode']).returnQuantity.sum()
    bought_articles = df.groupby(['sizeCode']).quantity.sum()
    size_return_prob = returned_articles / bought_articles
    return pd.Series(list(size_return_prob.loc[df.sizeCode]), index=df.index)


def customer_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    returned_articles = df.groupby(['customerID']).returnQuantity.sum()
    bought_articles = df.groupby(['customerID']).quantity.sum()
    customer_return_prob = returned_articles / bought_articles
    return pd.Series(list(customer_return_prob.loc[df.customerID]), index=df.index)


def product_group_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    returned_articles = df.groupby(['productGroup']).returnQuantity.sum()
    bought_articles = df.groupby(['productGroup']).quantity.sum()
    product_group_return_prob = returned_articles / bought_articles
    return pd.Series(list(product_group_return_prob.loc[df.productGroup]), index=df.index)


def same_article_surplus(df: pd.DataFrame) -> pd.DataFrame:
    article_group = df.groupby(['orderID', 'articleID']).quantity.sum()
    index = list(zip(df.orderID, df.articleID))
    return pd.Series(list(article_group.loc[index]) - df.quantity, index=df.index)


def same_article_same_size_surplus(df: pd.DataFrame) -> pd.DataFrame:
    article_size_group = df.groupby(['orderID', 'articleID', 'sizeCode']).quantity.sum()
    index = list(zip(df.orderID, df.articleID, df.sizeCode))
    return pd.Series(list(article_size_group.loc[index]) - df.quantity, index=df.index)


def same_article_same_color_surplus(df: pd.DataFrame) -> pd.DataFrame:
    article_size_group = df.groupby(['orderID', 'articleID', 'colorCode']).quantity.sum()
    index = list(zip(df.orderID, df.articleID, df.colorCode))
    return pd.Series(list(article_size_group.loc[index]) - df.quantity, index=df.index)


def total_order_share(df: pd.DataFrame) -> pd.DataFrame:
    order_prices = df.groupby(['orderID']).price.sum()
    return pd.Series(np.nan_to_num(df.price / list(order_prices.loc[df.orderID])), index=df.index)


def voucher_saving(df: pd.DataFrame) -> pd.DataFrame:
    order_prices = df.groupby(['orderID']).price.sum()
    voucher_amounts = df.groupby(['orderID']).voucherAmount.sum()
    savings = list(voucher_amounts.loc[df.orderID] / order_prices.loc[df.orderID])
    savings = np.nan_to_num(savings)
    return pd.Series(savings, index=df.index)


def date_to_season(date):
    spring = range(79, 177)  # 03/20
    summer = range(177, 266)  # 06/21
    fall = range(266, 356)  # 09/23
    if date.dayofyear in spring:
        return 2
    if date.dayofyear in summer:
        return 3
    if date.dayofyear in fall:
        return 4
    return 1


def total_day(date):
    return date.dayofyear if date.year == 2014 else date.dayofyear + 365


def binned_color_code(df: pd.DataFrame, deviations=1.0) -> pd.DataFrame:
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

    Returns
    -------
    pd.Series
        Series with a bin assigned to each row

    """

    def ret_prob(d: pd.Series) -> np.float64:
        return d.astype(bool).sum() / len(d)

    cc_ret_probs = df.groupby('colorCode')['returnQuantity'].apply(ret_prob)

    mean = cc_ret_probs.mean()
    diff = cc_ret_probs.std() * deviations

    mean_distances = cc_ret_probs.sub(mean).abs()

    bins = [COLOR_CODE_MIN]

    # iterate over colorCodes and respective mean distances to collect bins
    for cc, mean_distance in mean_distances.items():
        if mean_distance > diff:
            # add the colorCode as 1-sized bin (current cc and cc + 1)
            # if the last colorCode was added, don't add the current cc, only cc + 1.
            if bins[-1] != cc:
                bins.append(cc)
            bins.append(cc + 1)
    bins.append(COLOR_CODE_MAX + 1)

    cut = list(pd.cut(df.colorCode, bins, right=False))
    return pd.Series(cut, index=df.index)


def featuring(df: pd.DataFrame):
    """Incredibly descriptive and awesome method name
    """
    df = add_features(df)
    df = remove_features(df)
    return df
