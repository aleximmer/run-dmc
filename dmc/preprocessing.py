import pandas as pd
import numpy as np
import holidays

from dmc.features import dependent, independent


def apply_features(df: pd.DataFrame):
    """Add features and drop unused ones.
    """
    df = add_independent_features(df)
    df = add_dependent_features(df)
    df = remove_features(df)
    return df


def assert_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Define constraints here
    2. Enforce later
    """
    # Column values
    assert (df.quantity != 0).all()
    assert (df.quantity >= df.returnQuantity).all()

    # Column types
    assert df.orderDate.dtype == np.dtype('<M8[ns]')
    assert df.orderID.dtype == np.int
    assert df.articleID.dtype == np.int
    assert df.customerID.dtype == np.int
    assert df.voucherID.dtype == np.float


def enforce_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Drop data which doesn't comply with constraints
    Dropped rows would be """
    df = df[df.quantity > 0]
    df = df[df.quantity >= df.returnQuantity]
    # nans in these rows definitely have returnQuantity == 0
    df = df.dropna(subset=['voucherID', 'rrp', 'productGroup'])
    return df


def parse_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to float and integer types"""
    df.orderDate = pd.to_datetime(df.orderDate)
    df.orderID = df.orderID.apply(lambda x: x.replace('a', '')).astype(np.int)
    df.articleID = df.articleID.apply(lambda x: x.replace('i', '')).astype(np.int)
    df.customerID = df.customerID.apply(lambda x: x.replace('c', '')).astype(np.int)
    df.voucherID = df.voucherID.apply(lambda x: str(x).replace('v', '')).astype(np.float)
    df.voucherID = np.nan_to_num(df.voucherID)
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns which are either duplicate or will be added within our framework

    - date features since we create all possible of them using pandas
    - binary target is an option for benchmarking later
    - last six are dropped because of amateurish feature engineering
    """
    blacklist = ['id', 't_orderDate', 't_orderDateWOYear', 't_season', 't_dayOfWeek',
                 't_dayOfMonth', 't_isWeekend', 't_singleItemPrice_per_rrp', 't_atLeastOneReturned',
                 't_voucher_usedOnlyOnce_A', 't_voucher_stdDevDiscount_A', 't_voucher_OrderCount_A',
                 't_voucher_hasAbsoluteDiscountValue_A', 't_voucher_firstUsedDate_A',
                 't_voucher_lastUsedDate_A']
    for key in [k for k in blacklist if k in df.columns]:
        df = df.drop(key, 1)
    return df


def cleanse(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_columns(df)
    df = parse_strings(df)
    df = enforce_constraints(df)
    assert_constraints(df)
    return df


def add_independent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add returnQuantity independent features to DataFrame.
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
    df['orderTotalDay'] = df.orderDate.apply(independent.total_day)
    df['orderSeason'] = df.orderDate.apply(independent.date_to_season)
    df['orderIsOnGermanHoliday'] = df.orderDate.apply(lambda x: 1 if x in holidays.DE() else 0)
    df['surplusArticleQuantity'] = independent.same_article_surplus(df)
    df['surplusArticleSizeQuantity'] = independent.same_article_same_size_surplus(df)
    df['surplusArticleColorQuantity'] = independent.same_article_same_color_surplus(df)
    df['totalOrderShare'] = independent.total_order_share(df)
    df['voucherSavings'] = independent.voucher_saving(df)
    # df['voucherFirstUsedDate'] = pd.to_datetime(df.t_voucher_firstUsedDate_A).apply(total_day)
    # df['voucherLastUsedDate'] = pd.to_datetime(df.t_voucher_lastUsedDate_A).apply(total_day)
    df['customerAvgUnisize'] = df.t_customer_avgUnisize.astype(np.int)
    return df


def add_dependent_features(df: pd.DataFrame) -> pd.DataFrame:
    dependent.binned_color_code(df)
    dependent.color_return_probability(df)
    dependent.size_return_probability(df)
    dependent.customer_return_probability(df)
    dependent.product_group_return_probability(df)
    return df


def remove_features(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.drop('t_voucher_firstUsedDate_A', 1)
    # df = df.drop('t_voucher_lastUsedDate_A', 1)
    df = df.drop('t_customer_avgUnisize', 1)
    return df
