import pandas as pd
import numpy as np


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
    df['orderDayOfYear'] = df.orderDate.apply(lambda x: x.dayofyear)
    df['orderQuarter'] = df.orderDate.apply(lambda x: x.quarter)
    df['orderSeason'] = df.orderDate.apply(date_to_season)
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
    df['totalOrderShare'] = df.price / list(order_prices.loc[df.orderID])
    return df


def voucher_saving(df: pd.DataFrame) -> pd.DataFrame:
    order_prices = df.groupby(['orderID']).price.sum()
    voucher_amounts = df.groupby(['orderID']).voucherAmount.sum()
    df['voucherSavings'] = list(voucher_amounts.loc[df.orderID] / order_prices.loc[df.orderID])
    return df


def date_to_season(date):
    if date.month <= 3 and date.day <= 22:
        return 1
    if date.month <= 6 and date.day <= 22:
        return 2
    if date.month <= 9 and date.day <= 22:
        return 3
    if date.month <= 12 and date.day <= 22:
        return 4
    return 1


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    return df
