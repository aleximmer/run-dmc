import pandas as pd


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features"""
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new features to the DataFrame"""
    df = customer_return_probability(df)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    df = encode_features(df)
    return df


def customer_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    returned_articles = df.groupby(['customerID'])['returnQuantity'].sum()
    bought_articles = df.groupby(['customerID'])['quantity'].sum()
    customer_return_probs = returned_articles / bought_articles
    df['customerReturnProbs'] = customer_return_probs.loc[df['customerID']]
    return df

def group_return_probability(df: pd.DataFrame) -> pd.DataFrame:
    returned_articles = df.groupby(['productGroup'])['returnQuantity'].sum()
    bought_articles = df.groupby(['productGroup'])['quantity'].sum()
    group_return_probs = returned_articles / bought_articles
    df['groupReturnProbs'] = group_return_probs.loc[df['customerID']]
    return df
