import pandas as pd
import numpy as np
import holidays


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
    df['relativeSavings'] = (1 - df.productPrice / df.rrp)
    df.relativeSavings = df.relativeSavings.fillna(1.) # / 0. only when price == 0
    df['orderYear'] = df.orderDate.apply(lambda x: x.year)
    df['orderMonth'] = df.orderDate.apply(lambda x: x.month)
    df['orderDay'] = df.orderDate.apply(lambda x: x.day)
    df['orderWeekDay'] = df.orderDate.apply(lambda x: x.dayofweek)
    df['orderDayOfYear'] = df.orderDate.apply(lambda x: x.dayofyear)
    df['orderWeek'] = df.orderDate.apply(lambda x: x.week)
    df['orderWeekOfYear'] = df.orderDate.apply(lambda x: x.weekofyear)
    df['orderTotalDay'] = df.orderDate.apply(lambda x: x.dayofyear if x.year == 2014
                                             else x.dayofyear + 365)
    df['orderQuarter'] = df.orderDate.apply(lambda x: x.quarter)
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


def merge_features(df: pd.DataFrame, feature_dfs: list) -> pd.DataFrame:
    unique_keys = ['orderID', 'articleID', 'colorCode', 'sizeCode']
    for feature_df in feature_dfs:
        # Drop all columns which are in both DFs but not in original_keys
        left_keys = set(df.columns.values.tolist()) - set(unique_keys)
        right_keys = set(feature_df.columns.values.tolist()) - set(unique_keys)
        conflicting_keys = list(set(left_keys) & set(right_keys))
        feature_df.drop(conflicting_keys, inplace=True, axis=1)
        df = pd.merge(df, feature_df, how='left', on=unique_keys)
    return df
