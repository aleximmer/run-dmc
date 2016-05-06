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
