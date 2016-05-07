import pandas as pd
import numpy as np
import holidays


class SelectedFeatures:

    _whitelist = [
        'articleID', 'colorCode', 'customerID', 'deviceID', 'orderDate', 'orderID',
        'paymentMethod', 'price', 'productGroup', 'quantity', 'returnQuantity', 'rrp',
        'sizeCode', 't_article_availableColors', 't_article_availableSizes',
        't_article_boughtCountGlobal', 't_article_priceChangeSTD_A', 't_customer_avgUnisize',
        't_customer_boughtArticleCount', 't_customer_orderCount', 't_customer_voucherCount',
        't_isChristmas', 't_isGift', 't_isOneSize_A', 't_isTypeBelt', 't_isTypePants',
        't_isTypeTop', 't_isWeekend_A', 't_order_article_sameArticlesCount',
        't_order_article_sameArticlesCount_DiffColor',
        't_order_article_sameArticlesCount_DiffSize', 't_order_boughtArticleCount',
        't_order_boughtArticleTypeCount', 't_order_cheapestPrice', 't_order_duplicateCount',
        't_order_hasVoucher_A', 't_order_meanPrice', 't_order_mostExpensivePrice',
        't_order_priceStd_A', 't_order_sameArticlesCount',
        't_order_sameArticlesCount_DiffColor', 't_order_sameArticlesCount_DiffSize',
        't_order_totalPrice', 't_order_totalPrice_diff_voucherAmount',
        't_paymentWithFee_A', 't_reducedPaymentMethod_A', 't_singleItemPrice',
        't_singleItemPrice_diff_rrp', 't_sizeCodeNumerized', 't_ssv', 't_unisize',
        't_unisizeOffset', 't_voucher_is10PercentVoucher', 't_voucher_is15PercentVoucher',
        't_voucher_isGiftVoucher', 't_voucher_isValueVoucher', 't_voucher_usedCount_A',
        't_wsv', 'voucherAmount', 'voucherID'
    ]

    _blacklist = [
        'id', 't_orderDate', 't_orderDateWOYear', 't_season', 't_dayOfWeek',
        't_dayOfMonth', 't_isWeekend', 't_singleItemPrice_per_rrp', 't_atLeastOneReturned',
        't_voucher_usedOnlyOnce_A', 't_voucher_stdDevDiscount_A', 't_voucher_OrderCount_A',
        't_voucher_hasAbsoluteDiscountValue_A', 't_voucher_firstUsedDate_A',
        't_voucher_lastUsedDate_A'
    ]

    @classmethod
    def get_whitelist(cls):
        return set(cls._whitelist)

    @classmethod
    def get_blacklist(cls):
        return set(cls._blacklist)

    @classmethod
    def get_all_features(cls):
        return cls.get_whitelist().union(cls.get_blacklist())


"""Dependent features"""


def add_dependent_features(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    train, test = customer_return_probability(train, test)  # customerReturnProb
    train, test = size_return_probability(train, test)  # sizeReturnProb
    train, test = product_group_return_probability(train, test)  # productGroupReturnProb
    train, test = binned_color_code_return_probability(train, test)  # binnedColorCode
    return train, test


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


def apply_return_probs(train: pd.DataFrame, test: pd.DataFrame,
                       source_column: str, target_column: str) -> (pd.DataFrame, pd.DataFrame):
    """Group the values in the 'source column' of the training data and for each value calculate
    the return probability (cf. group_return_probability). Then add the 'target column' to training
    and test set containing the probability for each row. Unknown values, i.e. values present in
    test but not in training data, will have NaN as return probability.

    Parameters
    ----------
    train : pd.DataFrame
        Training data containing 'source column' and 'returnQuantity'
    test : pd.DataFrame
        Test data containing 'source column'
    source_column : str
        The column to group by. E.g., given 'customerID' the return probabilities for each customer
        will be calculated
    target_column : str
        The name of the column to put in the probabilities.

    Returns
    -------
    pd.DataFrame
        Training data with target column added
    pd.DataFrame
        Test data with target column added
    """
    ret_probs = train.groupby(source_column)['returnQuantity'].apply(group_return_probability)
    train[target_column] = ret_probs.reindex(train[source_column]).values
    test[target_column] = ret_probs.reindex(test[source_column]).values
    return train, test


def customer_return_probability(train, test) -> (pd.DataFrame, pd.DataFrame):
    """Calculate likelihood of a specific customer to return a product.
    """
    return apply_return_probs(train, test, 'customerID', 'customerReturnProb')


def size_return_probability(train, test) -> (pd.DataFrame, pd.DataFrame):
    """Calculate likelihood of an order with a specific sizeCode to result in a return.
    """
    return apply_return_probs(train, test, 'sizeCode', 'sizeReturnProb')


def product_group_return_probability(train, test) -> (pd.DataFrame, pd.DataFrame):
    """Calculate likelihood of an order with a specific productGroup to result in a return.
    """
    return apply_return_probs(train, test, 'productGroup', 'productGroupReturnProb')


def binned_color_code_return_probability(train, test, deviations=1.0) -> (
        pd.DataFrame, pd.DataFrame):
    """Bin the colorCode column in training and test using return probabilities of training data.

    Notes
    -----
    This is to deal with unknown colorCodes (CC's) in the target set by binning the CC range.
    The binning considers outlier CC's by keeping them separate, 1-sized bins.
    Outliers are CC's whose return probability is over one standard deviation away from the mean.
    For our training data (mean: 0.548, std: 0.114) colorCode c is an is an outlier if

        retProb(c) < 0.434 || 0.662 < retProb(c), given deviations = 1.

    Parameters
    ----------
    train : pd.DataFrame
    test : pd.DataFrame
        See apply_return_probs
    deviations : float
        Number of standard deviations a return probability has to differ from the mean to be
        considered an outlier.
    """
    color_code_min = 0
    color_code_max = 9999

    # Calculate return probability for each colorCode
    color_code_ret_probs = (train
                            .groupby('colorCode')['returnQuantity']
                            .apply(group_return_probability))

    # Reindex those values to resemble to distribution in the training set
    row_ret_probs = color_code_ret_probs.reindex(train['colorCode'])

    # Calculate mean and minimum mean distance
    mean = row_ret_probs.mean()
    diff = row_ret_probs.std() * deviations
    mean_distances = color_code_ret_probs.sub(mean).abs()

    # iterate over colorCodes and respective mean distances to collect bins
    bins = [color_code_min]
    for cc, mean_distance in mean_distances.items():
        if mean_distance > diff:
            # add the colorCode as 1-sized bin (current cc and cc + 1)
            # if the last colorCode was added, don't add the current cc, only cc + 1.
            if bins[-1] != cc:
                bins.append(cc)
            bins.append(cc + 1)
    bins.append(color_code_max + 1)

    # Assign bins to each row in test and training data
    train['binnedColorCode'] = pd.cut(train['colorCode'], bins, right=False, labels=False)
    test['binnedColorCode'] = pd.cut(test['colorCode'], bins, right=False, labels=False)

    train, test = apply_return_probs(train, test, 'binnedColorCode', 'colorReturnProb')
    # Test set colorCodes that are bigger than any colorCode in the training set fall into a
    # category that has no returnProbability. Impute that bin with the mean retProb.
    test['colorReturnProb'] = test['colorReturnProb'].fillna(mean)
    return train, test


"""Independent features"""


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
    df['orderTotalDay'] = df.orderDate.apply(total_day)
    df['orderSeason'] = df.orderDate.apply(date_to_season)
    df['orderIsOnGermanHoliday'] = df.orderDate.apply(lambda x: 1 if x in holidays.DE() else 0)
    df['surplusArticleQuantity'] = same_article_surplus(df)
    df['surplusArticleSizeQuantity'] = same_article_same_size_surplus(df)
    df['surplusArticleColorQuantity'] = same_article_same_color_surplus(df)
    df['totalOrderShare'] = total_order_share(df)
    df['voucherSavings'] = voucher_saving(df)
    # df['voucherFirstUsedDate'] = pd.to_datetime(df.t_voucher_firstUsedDate_A).apply(total_day)
    # df['voucherLastUsedDate'] = pd.to_datetime(df.t_voucher_lastUsedDate_A).apply(total_day)
    df['customerAvgUnisize'] = df.t_customer_avgUnisize.astype(np.int)
    return df


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
