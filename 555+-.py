import ccxt
import logging
import requests
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import threading

api_key = " "
api_secret = " "

exchange = ccxt.bybit({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

symbol_xrp = 'XRP/USDT'
short_window = 10
long_window = 30
rsi_period = 14
trading_fee_percentage = 0.001  # комиссия составляет 0.1% на каждую сделку

logging.basicConfig(filename='trading_bot.log', level=logging.INFO)

telegram_token = ' '
telegram_chat_id = ' '

trading_active = True
trade_history = []

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    data = {'chat_id': telegram_chat_id, 'text': message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

def get_current_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Error fetching ticker for {symbol}: {e}")
        return None

def get_available_balance(asset):
    try:
        balance = exchange.fetch_balance()
        return balance['total'].get(asset, 0)
    except Exception as e:
        logging.error(f"Error fetching balance for {asset}: {e}")
        return 0

def get_moving_average(symbol, period):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=period)
        closes = [x[4] for x in ohlcv]
        return np.mean(closes[-period:])
    except Exception as e:
        logging.error(f"Error fetching moving average for {symbol}: {e}")
        return None

def calculate_rsi(symbol, period):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=period + 1)
        closes = np.array([x[4] for x in ohlcv])
        price_changes = np.diff(closes)

        gain = np.where(price_changes > 0, price_changes, 0)
        loss = np.where(price_changes < 0, -price_changes, 0)

        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        return rsi
    except Exception as e:
        logging.error(f"Error calculating RSI for {symbol}: {e}")
        return None

def calculate_bollinger_bands(symbol, period, std_dev):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=period)
        closes = np.array([x[4] for x in ohlcv])
        sma = np.mean(closes)
        rolling_std = np.std(closes)
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, lower_band
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands for {symbol}: {e}")
        return None, None

def calculate_macd(symbol, slow_period=26, fast_period=12, signal_period=9):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=slow_period + signal_period)
        closes = np.array([x[4] for x in ohlcv])
        fast_ema = pd.Series(closes).ewm(span=fast_period, adjust=False).mean()
        slow_ema = pd.Series(closes).ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line.iloc[-1], signal_line.iloc[-1]
    except Exception as e:
        logging.error(f"Error calculating MACD for {symbol}: {e}")
        return None, None

def place_order(order_type, symbol, price, amount):
    available_usdt_balance = get_available_balance('USDT')
    available_xrp_balance = get_available_balance('XRP')

    # Проверка на доступный баланс перед размещением ордера
    if order_type == 'buy':
        cost = amount * price + (amount * price * trading_fee_percentage)
        if cost > available_usdt_balance:
            logging.error("Not enough USDT to place the buy order.")
            return
    elif order_type == 'sell':
        if amount > available_xrp_balance:
            logging.error("Not enough XRP to place the sell order.")
            return

    try:
        fee = price * amount * trading_fee_percentage
        if order_type == 'buy':
            order = exchange.create_limit_buy_order(symbol, amount, price)
            send_telegram_message(f'Auto Buy (XRP) at {price}. Order amount: {amount}')
        elif order_type == 'sell':
            order = exchange.create_limit_sell_order(symbol, amount, price)
            send_telegram_message(f'Auto Sell (XRP) at {price}. Order amount: {amount}')

        trade_history.append({
            'type': order_type,
            'symbol': symbol,
            'price': price,
            'amount': amount,
            'fee': fee,
            'timestamp': time.time()
        })

    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")

async def trade_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    history_message = "История сделок:\n"
    total_fees = 0
    for trade in trade_history:
        total_fees += trade['fee']
        history_message += (f"{trade['timestamp']}: {trade['type'].capitalize()} {trade['amount']} "
                            f"по цене {trade['price']} {trade['symbol']} с комиссией {trade['fee']}\n")
    
    history_message += f"Всего уплачено комиссий: {total_fees}\n"
    await update.message.reply_text(history_message)

async def plot_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not trade_history:
        await update.message.reply_text("История сделок пуста.")
        return

    prices = [trade['price'] for trade in trade_history]
    timestamps = [trade['timestamp'] for trade in trade_history]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, prices, marker='o')
    plt.title('История цен сделок')
    plt.xlabel('Время')
    plt.ylabel('Цена')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    await update.message.reply_photo(photo=buf)    
    
def get_available_balance(asset):
    balance_info = exchange.fetch_balance()
    return balance_info[asset]['free']

def auto_trade():
    global trading_active
    try:
        exchange.load_markets()
        market_info = exchange.markets.get(symbol_xrp)

        if not market_info:
            logging.error(f"Market info for {symbol_xrp} not found.")
            return

        min_order_size = market_info['limits']['amount']['min']
    except Exception as e:
        logging.error(f"Error loading market information: {e}")
        return

    while True:
        if trading_active:
            current_price_xrp = get_current_price(symbol_xrp)
            available_usdt_balance = get_available_balance('USDT')
            available_xrp_balance = get_available_balance('XRP')

            if current_price_xrp is not None:
                amount_to_buy = 0.20 * available_usdt_balance / current_price_xrp

                if amount_to_buy < min_order_size:
                    amount_to_buy = min_order_size

                if amount_to_buy * current_price_xrp <= available_usdt_balance:
                    short_ma_xrp = get_moving_average(symbol_xrp, short_window)
                    long_ma_xrp = get_moving_average(symbol_xrp, long_window)
                    rsi_xrp = calculate_rsi(symbol_xrp, rsi_period)
                    upper_band, lower_band = calculate_bollinger_bands(symbol_xrp, 20, 2)
                    macd_line, signal_line = calculate_macd(symbol_xrp)

                    buy_condition = (
                        current_price_xrp <= short_ma_xrp and
                        rsi_xrp < 30 and
                        current_price_xrp <= lower_band and
                        macd_line > signal_line
                    )

                    sell_condition = (
                        current_price_xrp >= long_ma_xrp and
                        available_xrp_balance > 0 and
                        rsi_xrp > 70 and
                        current_price_xrp >= upper_band and
                        macd_line < signal_line
                    )

                    if buy_condition:
                        try:
                            place_order('buy', symbol_xrp, current_price_xrp, amount_to_buy)
                        except Exception as e:
                            logging.error(f"Error placing buy order for {symbol_xrp}: {e}")

                    elif sell_condition:
                        try:
                            place_order('sell', symbol_xrp, current_price_xrp, available_xrp_balance)
                        except Exception as e:
                            logging.error(f"Error placing sell order for {symbol_xrp}: {e}")
                else:
                    logging.error("Not enough balance to place the order.")

        time.sleep(60)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global trading_active
    trading_active = True
    await update.message.reply_text('Trading bot started!')

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global trading_active
    trading_active = False
    await update.message.reply_text('Trading bot stopped!')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    available_balance = get_available_balance('USDT')
    updated_current_xrp_balance = get_available_balance('XRP')
    status_message = (f'Trading Active: {trading_active}\n'
                      f'Available USDT Balance: {available_balance}\n'
                      f'Current XRP Balance: {updated_current_xrp_balance}')
    await update.message.reply_text(status_message)

def main():
    application = ApplicationBuilder().token(telegram_token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("history", trade_history_command))
    application.add_handler(CommandHandler("plot", plot_history))

    trading_thread = threading.Thread(target=auto_trade, daemon=True)
    trading_thread.start()

    application.run_polling()

if __name__ == '__main__':
    main()