import pandas as pd


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features"""
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new features to the DataFrame"""
    df['productPrice'] = df.price / df.quantity
    df['totalSavings'] = df.rrp - df.productPrice
    df['relativeSavings'] = 1 - df.productPrice / df.rrp
    df['orderYear'] = df.orderDate.apply(lambda x: x.year)
    df['orderMonth'] = df.orderDate.apply(lambda x: x.month)
    df['orderDay'] = df.orderDate.apply(lambda x: x.day)
    df['orderWeekDay'] = df.orderDate.apply(lambda x: x.dayofweek)
    df['orderDayOfYear'] = df.orderDate.apply(lambda x: x.dayofyear)
    df['orderWeek'] = df.orderDate.apply(lambda x: x.week)
    df['orderWeekOfYear'] = df.orderDate.apply(lambda x: x.weekofyear)
    df['orderYearDay'] = df.orderDate.apply(lambda x: x.timetuple().tm_yday)
    df['orderQuarter'] = df.orderDate.apply(lambda x: x.quarter)
    df['orderSeason'] = df.orderDate.apply(date_to_season)
    df = customer_return_probability(df)
    df = same_article_surplus(df)
    df = same_article_same_size_surplus(df)
    df = same_article_same_color_surplus(df)
    return df


def customer_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    customer_return_probs = (df.groupby(['customerID']).returnQuantity.sum() /
                             df.groupby(['customerID']).quantity.sum())
    df['customerReturnProbs'] = customer_return_probs.loc[df.customerID]
    return df


def same_article_surplus(df: pd.DataFrame) -> pd.DataFrame:
    article_group = df.groupby(['orderID', 'articleID'])['quantity'].sum()
    index = list(zip(df.orderID, df.articleID))
    df['surplusArticleQuantity'] = list(article_group.loc[index]) - df.quantity
    return df


def same_article_same_size_surplus(df: pd.DataFrame) -> pd.DataFrame:
    article_size_group = df.groupby(['orderID', 'articleID', 'sizeCode'])['quantity'].sum()
    index = list(zip(df.orderID, df.articleID, df.sizeCode))
    df['surplusArticleSizeQuantity'] = list(article_size_group.loc[index]) - df.quantity
    return df


def same_article_same_color_surplus(df: pd.DataFrame) -> pd.DataFrame:
    article_size_group = df.groupby(['orderID', 'articleID', 'colorCode'])['quantity'].sum()
    index = list(zip(df.orderID, df.articleID, df.colorCode))
    df['surplusArticleColorQuantity'] = list(article_size_group.loc[index]) - df.quantity
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
    df = encode_features(df)
    return df
