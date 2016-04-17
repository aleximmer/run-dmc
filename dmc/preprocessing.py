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
    df['orderQuarter'] = df.orderDate.apply(lambda x: x.quarter)
    df = customer_return_probability(df)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    df = encode_features(df)
    return df


def customer_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    customer_return_probs = (df.groupby(['customerID'])['returnQuantity'].sum() /
                             df.groupby(['customerID'])['quantity'].sum())
    df['customerReturnProbs'] = customer_return_probs.loc[df['customerID']]
    return df


def season(df: pd.DataFrame) -> pd.DataFrame:
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
    df['season'] = df.orderDate.copy().apply(date_to_season)
    return df
