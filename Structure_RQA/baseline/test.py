import talib

# 初始化函数
def init(context):
    # 设置股票池
    context.s1 = "000001.XSHE"
    context.s2 = "601988.XSHG"
    context.s3 = "000068.XSHE"
    context.stocks = [context.s1, context.s2, context.s3]

    # RSI参数
    context.TIME_PERIOD = 14
    context.HIGH_RSI = 85
    context.LOW_RSI = 30
    context.ORDER_PERCENT = 0.3


# 每个bar数据到来时调用
def handle_bar(context, bar_dict):
    for stock in context.stocks:
        # 获取最近 TIME_PERIOD + 1 个交易日的收盘价
        prices = history_bars(stock, context.TIME_PERIOD + 1, '1d', 'close')

        # 如果数据不足，跳过
        if prices is None or len(prices) < context.TIME_PERIOD + 1:
            continue

        # 计算 RSI
        rsi_series = talib.RSI(prices, timeperiod=context.TIME_PERIOD)
        yesterday_rsi = rsi_series[-1]  # 使用“昨天”的 RSI 值，避免未来数据泄露

        # 当前持仓数量
        cur_position = get_position(stock).quantity

        # 可用现金的指定比例
        target_available_cash = context.portfolio.cash * context.ORDER_PERCENT

        # 卖出信号：RSI 高于上限且有持仓
        if yesterday_rsi > context.HIGH_RSI and cur_position > 0:
            order_target_value(stock, 0)

        # 买入信号：RSI 低于下限
        elif yesterday_rsi < context.LOW_RSI:
            logger.info(f"{stock} RSI={yesterday_rsi:.2f}, buying with cash: {target_available_cash:.2f}")
            order_value(stock, target_available_cash)
