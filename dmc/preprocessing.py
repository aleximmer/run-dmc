import pandas as pd
import numpy as np
import holidays


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new features to the DataFrame"""
    df['productPrice'] = df.price / df.quantity
    df['totalSavings'] = df.rrp - df.productPrice
    df['relativeSavings'] = np.nan_to_num(1 - df.productPrice / df.rrp)
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
    df = color_return_probability(df)
    df = size_return_probability(df)
    df = customer_return_probability(df)
    df = product_group_return_probability(df)
    df = same_article_surplus(df)
    df = same_article_same_size_surplus(df)
    df = same_article_same_color_surplus(df)
    df = total_order_share(df)
    df = voucher_saving(df)
    return df


def color_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    returned_articles = df.groupby(['colorCode']).returnQuantity.sum()
    bought_articles = df.groupby(['colorCode']).quantity.sum()
    color_return_prob = returned_articles / bought_articles
    df['colorReturnProb'] = list(color_return_prob.loc[df.colorCode])
    return df


def size_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    returned_articles = df.groupby(['sizeCode']).returnQuantity.sum()
    bought_articles = df.groupby(['sizeCode']).quantity.sum()
    size_return_prob = returned_articles / bought_articles
    df['sizeReturnProb'] = list(size_return_prob.loc[df.sizeCode])
    return df


def customer_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    returned_articles = df.groupby(['customerID']).returnQuantity.sum()
    bought_articles = df.groupby(['customerID']).quantity.sum()
    customer_return_prob = returned_articles / bought_articles
    df['customerReturnProb'] = list(customer_return_prob.loc[df.customerID])
    return df


def product_group_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    returned_articles = df.groupby(['productGroup']).returnQuantity.sum()
    bought_articles = df.groupby(['productGroup']).quantity.sum()
    product_group_return_prob = returned_articles / bought_articles
    df['productGroupReturnProb'] = list(product_group_return_prob.loc[df.productGroup])
    return df


def same_article_surplus(df: pd.DataFrame) -> pd.DataFrame:
    article_group = df.groupby(['orderID', 'articleID']).quantity.sum()
    index = list(zip(df.orderID, df.articleID))
    df['surplusArticleQuantity'] = list(article_group.loc[index]) - df.quantity
    return df


def same_article_same_size_surplus(df: pd.DataFrame) -> pd.DataFrame:
    article_size_group = df.groupby(['orderID', 'articleID', 'sizeCode']).quantity.sum()
    index = list(zip(df.orderID, df.articleID, df.sizeCode))
    df['surplusArticleSizeQuantity'] = list(article_size_group.loc[index]) - df.quantity
    return df


def same_article_same_color_surplus(df: pd.DataFrame) -> pd.DataFrame:
    article_size_group = df.groupby(['orderID', 'articleID', 'colorCode']).quantity.sum()
    index = list(zip(df.orderID, df.articleID, df.colorCode))
    df['surplusArticleColorQuantity'] = list(article_size_group.loc[index]) - df.quantity
    return df


def total_order_share(df: pd.DataFrame) -> pd.DataFrame:
    order_prices = df.groupby(['orderID']).price.sum()
    df['totalOrderShare'] = np.nan_to_num(df.price / list(order_prices.loc[df.orderID]))
    return df


def voucher_saving(df: pd.DataFrame) -> pd.DataFrame:
    order_prices = df.groupby(['orderID']).price.sum()
    voucher_amounts = df.groupby(['orderID']).voucherAmount.sum()
    df['voucherSavings'] = list(voucher_amounts.loc[df.orderID] / order_prices.loc[df.orderID])
    df.voucherSavings = np.nan_to_num(df.voucherSavings)
    return df


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
