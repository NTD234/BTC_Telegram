"""
BotTrade Telegram - Binance Version (FINAL)
Cấu trúc code được tổ chức rõ ràng theo từng mục chức năng
"""

import os
import logging
import pandas as pd
import pandas_ta as ta
import asyncio
from datetime import datetime, timezone, timedelta
from binance.client import Client
from telegram import Bot

# ==============================================================================
# CONFIGURATION
# ==============================================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "15"))
SYMBOL = "BTCUSDT"
VN_TZ = timezone(timedelta(hours=7))

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Binance client
binance = Client()

def get_vn_time():
    """Lấy thời gian Việt Nam hiện tại"""
    return datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M")


# ==============================================================================
# TRADING PARAMETERS
# ==============================================================================

# Score thresholds
MIN_SCORE = 50           # Score tối thiểu cho LONG
MIN_SCORE_SHORT = 45     # Score tối thiểu cho SHORT
MIN_REV_LONG_SHORT = 75  # Score tối thiểu cho LONG -> SHORT
MIN_REV_SHORT_LONG = 70  # Score tối thiểu cho SHORT -> LONG

# SL/TP parameters
SL_MULTIPLIER = 1.5      # ATR × 1.5 cho SL (LONG)
MIN_SL_PCT = 0.01        # Minimum SL 1% cho LONG
SL_SHORT_PCT = 0.01      # SL cố định 1% cho SHORT
TP_MULTIPLIER = 1.9      # TP = SL × 1.9 (optimized)
TRAILING_MULTIPLIER = 0  # Trailing distance = ATR × 0.8

# ADX thresholds
ADX_TREND = 20           # ADX > 20: Có xu hướng, cấm trade ngược trend
ADX_STRONG = 30          # ADX > 30: Xu hướng mạnh, chỉ trade theo trend


# ==============================================================================
# DATA FETCHING
# ==============================================================================

def get_data(symbol=SYMBOL, limit=500, max_retries=5):
    """
    Lấy dữ liệu M15 và H1 từ Binance với retry logic
    
    Returns:
        df_m15: DataFrame M15
        df_h1: DataFrame H1 (resampled từ M15)
    """
    for attempt in range(max_retries):
        try:
            klines = binance.futures_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_15MINUTE,
                limit=limit
            )

            df = pd.DataFrame(klines, columns=[
                'time','open','high','low','close','volume',
                '_','_','_','_','_','_'
            ])

            df = df[['time','open','high','low','close','volume']]
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Asia/Ho_Chi_Minh').dt.tz_localize(None)
            df.set_index('time', inplace=True)

            df_h1 = df.resample('1h').agg({
                'open':'first','high':'max','low':'min','close':'last','volume':'sum'
            }).dropna()

            return df, df_h1
            
        except Exception as e:
            wait_time = (2 ** attempt) * 30
            logger.warning(f"Lỗi Binance API (lần {attempt + 1}/{max_retries}): {e}")
            logger.info(f"Chờ {wait_time}s trước khi thử lại...")
            import time
            time.sleep(wait_time)
    
    raise Exception(f"Không thể lấy dữ liệu sau {max_retries} lần thử")


# ==============================================================================
# INDICATOR CALCULATIONS
# ==============================================================================

def calculate_indicators_h1(df_h1):
    """
    Tính toán các chỉ báo trên timeframe H1
    
    Indicators:
    - Stochastic (16-16-8) + slope + count
    - ADX + DI + slope
    - MACD histogram
    - RSI + slope
    - ATR
    - TSI (Trend Strength Index)
    """
    # Stochastic H1
    stoch = ta.stoch(df_h1['high'], df_h1['low'], df_h1['close'], k=16, smooth_k=16, d=8)
    df_h1['stoch_k'] = stoch.iloc[:, 0]
    df_h1['stoch_d'] = stoch.iloc[:, 1]
    df_h1['stoch_slope'] = df_h1['stoch_k'].diff()
    df_h1['stoch_neg_count'] = (df_h1['stoch_slope'].shift(1) < 0).rolling(4).sum()
    df_h1['stoch_pos_count'] = (df_h1['stoch_slope'].shift(1) > 0).rolling(4).sum()

    # ADX + DI
    adx_data = ta.adx(df_h1['high'], df_h1['low'], df_h1['close'])
    df_h1['adx'] = adx_data['ADX_14']
    df_h1['adx_slope'] = df_h1['adx'].diff()
    df_h1['plus_di'] = adx_data['DMP_14']
    df_h1['minus_di'] = adx_data['DMN_14']
    
    # MACD
    macd = ta.macd(df_h1['close'])
    df_h1['macd_hist'] = macd.iloc[:, 2]
    
    # RSI
    df_h1['rsi'] = ta.rsi(df_h1['close'])
    df_h1['rsi_slope'] = df_h1['rsi'].diff()
    
    # ATR
    df_h1['atr_h1'] = ta.atr(df_h1['high'], df_h1['low'], df_h1['close'])
    
    # TSI H1 (Trend Strength Index = Pearson Correlation)
    def calc_tsi_h1(prices):
        import numpy as np
        if len(prices) < 2:
            return 0
        x = np.arange(len(prices))
        y = np.array(prices)
        n = len(prices)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        sum_y2 = np.sum(y ** 2)
        numerator = n * sum_xy - sum_x * sum_y
        denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        if denominator == 0:
            return 0
        return numerator / denominator
    
    df_h1['tsi_h1'] = df_h1['close'].rolling(16).apply(calc_tsi_h1, raw=True)
    df_h1['tsi_h1_slope'] = df_h1['tsi_h1'].diff()
    
    return df_h1


def calculate_indicators_m15(df):
    """
    Tính toán các chỉ báo trên timeframe M15
    
    Indicators:
    - Stochastic M15 (16-16-8) + slope
    - KAMA (10-2-20) + slope
    - TSI (Trend Strength Index)
    - Donchian Mid
    - MFI
    - ATR + ATR average
    - VWAP
    - Volatility %
    - ADX M15
    """
    # Stochastic M15
    stoch_m15 = ta.stoch(df['high'], df['low'], df['close'], k=16, smooth_k=16, d=8)
    df['stoch_k_m15'] = stoch_m15.iloc[:, 0]
    df['stoch_d_m15'] = stoch_m15.iloc[:, 1]
    df['stoch_slope_m15'] = df['stoch_k_m15'].diff()

    # KAMA
    df['kama'] = ta.kama(df['close'], length=10, fast=2, slow=20)
    df['kama_slope'] = df['kama'].diff()

    # TSI M15 (Trend Strength Index = Pearson Correlation)
    def calc_tsi(prices):
        import numpy as np
        if len(prices) < 2:
            return 0
        x = np.arange(len(prices))
        y = np.array(prices)
        n = len(prices)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        sum_y2 = np.sum(y ** 2)
        numerator = n * sum_xy - sum_x * sum_y
        denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        if denominator == 0:
            return 0
        return numerator / denominator
    
    df['tsi'] = df['close'].rolling(16).apply(calc_tsi, raw=True)

    # Donchian
    dc = ta.donchian(df['high'], df['low'])
    df['dc_mid'] = dc.iloc[:, 1]

    # MFI
    df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
    
    # ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'])
    df['atr_avg'] = df['atr'].rolling(50).mean()
    
    # VWAP
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    
    # Volatility
    df['volatility_pct'] = (df['high'] - df['low']) / df['close'] * 100
    df['max_volatility_2'] = df['volatility_pct'].shift(1).rolling(2).max()
    
    # ADX M15
    adx_m15 = ta.adx(df['high'], df['low'], df['close'])
    df['adx_m15'] = adx_m15['ADX_14']
    
    # BBW (Bollinger Band Width) - Sideway detection
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_lower"] = bb.iloc[:, 0]
    df["bb_middle"] = bb.iloc[:, 1]
    df["bb_upper"] = bb.iloc[:, 2]
    df["bbw"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    
    return df


def calculate_indicators(df_m15, df_h1):
    """
    Tính toán tất cả indicators và merge H1 vào M15
    
    Returns:
        DataFrame M15 với tất cả indicators (H1 + M15)
    """
    # Tính indicators cho từng timeframe
    df_h1 = calculate_indicators_h1(df_h1)
    df_m15 = calculate_indicators_m15(df_m15)
    
    # Join H1 indicators vào M15
    h1_cols = [
        'stoch_k', 'stoch_d', 'stoch_slope', 'stoch_neg_count', 'stoch_pos_count',
        'adx', 'adx_slope', 'plus_di', 'minus_di',
        'macd_hist', 'rsi', 'rsi_slope', 'atr_h1', 'tsi_h1', 'tsi_h1_slope'
    ]
    df = df_m15.join(df_h1[h1_cols], how='left').ffill()
    
    # Drop rows thiếu indicators quan trọng
    essential_cols = ['close', 'stoch_k', 'stoch_slope', 'kama', 'atr']
    return df.dropna(subset=essential_cols)


# ==============================================================================
# TRADING LOGIC - SIGNAL ANALYSIS
# ==============================================================================

def get_trend_direction(row):
    """
    Xác định hướng xu hướng dựa trên +DI và -DI
    
    Returns:
        'UP', 'DOWN', hoặc 'NEUTRAL'
    """
    if pd.isna(row.get('plus_di')) or pd.isna(row.get('minus_di')):
        return 'NEUTRAL'
    if row['plus_di'] > row['minus_di']:
        return 'UP'
    elif row['minus_di'] > row['plus_di']:
        return 'DOWN'
    return 'NEUTRAL'


# ==============================================================================
# PENDING SIGNAL CLASS
# ==============================================================================

class PendingSignal:
    """
    Class để quản lý pending signals khi BBW quá thấp
    """
    def __init__(self):
        self.direction = None
        self.score = None
        self.entry_time = None
        self.bbw_threshold = 0.0025  # BBW threshold (synced with Backtest)
    
    def set_pending(self, direction, score, entry_time):
        """Lưu signal vào pending state"""
        self.direction = direction
        self.score = score
        self.entry_time = entry_time
    
    def clear(self):
        """Clear pending signal"""
        self.direction = None
        self.score = None
        self.entry_time = None
    
    def has_pending(self):
        """Check if có pending signal"""
        return self.direction is not None


def analyze_signal(row):
    """
    Phân tích tín hiệu trading từ một row dữ liệu
    
    Returns:
        tuple: (direction, score, k_high, is_pending)
        - direction: 'LONG', 'SHORT', or None
        - score: int (0-100)
        - k_high: bool (K > 70 bonus for SHORT)
        - is_pending: bool (True if BBW too low)
    
    Logic:
    1. Entry signal: Stoch H1 + KAMA
    2. ADX filters (trend direction + strength)
    3. TSI filters (extreme conditions)
    4. Volatility filter
    5. BBW filter (pending if too low)
    5. Stoch H1 + TSI H1 extreme filter
    6. Scoring system
    
    Returns:
        (direction, score, k_high)
        - direction: 'LONG', 'SHORT', hoặc None
        - score: 0-100
        - k_high: True nếu SHORT và K > 70
    """
    if pd.isna(row.get("stoch_k")) or pd.isna(row.get("stoch_slope")):
        return None, 0, False, False

    score = 0
    direction = None
    k = row['stoch_k']
    k_slope = row['stoch_slope']
    adx = row.get("adx", 0) if not pd.isna(row.get("adx")) else 0
    trend = get_trend_direction(row)

    # ===== BƯỚC 1: Entry Signal (Stoch H1 + KAMA) =====
    kama_slope = row.get('kama_slope', 0) if not pd.isna(row.get('kama_slope')) else 0
    stoch_neg_count = row.get('stoch_neg_count', 0) if not pd.isna(row.get('stoch_neg_count')) else 0
    stoch_pos_count = row.get('stoch_pos_count', 0) if not pd.isna(row.get('stoch_pos_count')) else 0
    
    # LONG: K tăng + ít nhất 2/4 nến trước giảm + close>KAMA + KAMA không dốc xuống
    if k_slope > 0 and stoch_neg_count >= 2:
        if not pd.isna(row.get("kama")) and row['close'] > row['kama'] and kama_slope >= 0:
            direction = "LONG"
    
    # SHORT: K giảm + ít nhất 2/4 nến trước tăng + close<KAMA + KAMA không dốc lên
    elif k_slope < 0 and stoch_pos_count >= 2:
        if not pd.isna(row.get("kama")) and row['close'] < row['kama'] and kama_slope <= 0:
            direction = "SHORT"
    
    if direction is None:
        return None, 0, False, False

    # ===== BƯỚC 2: ADX Filters =====
    # ADX > 30: Trend mạnh → BẮT BUỘC theo trend
    if adx > ADX_STRONG:
        if direction == "LONG" and trend != "UP":
            return None, 0, False, False
        if direction == "SHORT" and trend != "DOWN":
            return None, 0, False, False
    
    # ADX > 20: Có trend → Cấm trade ngược trend
    elif adx > ADX_TREND:
        if direction == "LONG" and trend == "DOWN":
            return None, 0, False, False
        if direction == "SHORT" and trend == "UP":
            return None, 0, False, False
        # SHORT chỉ khi ADX đang giảm
        adx_slope = row.get('adx_slope', 0) if not pd.isna(row.get('adx_slope')) else 0
        if direction == "SHORT" and adx_slope >= 0:
            return None, 0, False, False

    # ADX M15 filter: Chỉ trade khi 10 < ADX M15 < 50
    adx_m15 = row.get('adx_m15', 0) if not pd.isna(row.get('adx_m15')) else 0
    if adx_m15 > 50 or adx_m15 < 10:
        return None, 0, False, False

    # ===== BƯỚC 3: TSI Filters =====
    tsi = row.get('tsi', 0) if not pd.isna(row.get('tsi')) else 0
    if direction == "SHORT" and tsi < -0.55:  # Quá oversold
        return None, 0, False, False
    if direction == "LONG" and tsi > 0.55:    # Quá overbought
        return None, 0, False, False

    # ===== BƯỚC 4: Volatility Filter =====
    max_vol = row.get('max_volatility_2', 0) if not pd.isna(row.get('max_volatility_2')) else 0
    if max_vol > 0.9:
        return None, 0, False, False

    # ===== BƯỚC 5: Stoch H1 + TSI H1 Extreme Filter =====
    k_h1 = row.get('stoch_k', 50) if not pd.isna(row.get('stoch_k')) else 50
    tsi_h1 = row.get('tsi_h1', 0) if not pd.isna(row.get('tsi_h1')) else 0
    
    # Không LONG khi Stoch H1 > 75 VÀ TSI H1 > 0.9 (quá overbought)
    if direction == "LONG" and k_h1 > 75 and tsi_h1 > 0.9:
        return None, 0, False, False
    
    # Không SHORT khi Stoch H1 < 25 VÀ TSI H1 < -0.9 (quá oversold)
    if direction == "SHORT" and k_h1 < 25 and tsi_h1 < -0.9:
        return None, 0, False, False

    # ===== BƯỚC 6: Scoring System =====
    k_m15 = row.get('stoch_k_m15', 50)
    d_m15 = row.get('stoch_d_m15', 50)
    k_slope_m15 = row.get('stoch_slope_m15', 0) if not pd.isna(row.get('stoch_slope_m15')) else 0

    if direction == "LONG":
        if k_m15 > d_m15 and k_slope_m15 > 0: score += 20
        if not pd.isna(row.get("rsi_slope")) and row.get("rsi_slope", 0) > 0 and row.get("rsi", 50) < 50: score += 20
        if not pd.isna(row.get("macd_hist")) and row.get("macd_hist", 0) > 0: score += 15
        if not pd.isna(row.get("dc_mid")) and row['close'] < row['dc_mid']: score += 15
        if not pd.isna(row.get("mfi")) and row.get("mfi", 50) > 30: score += 15
        if not pd.isna(row.get("atr")) and not pd.isna(row.get("atr_avg")) and row['atr'] > row['atr_avg']: score += 15

    if direction == "SHORT":
        if k_m15 < d_m15 and k_slope_m15 < 0: score += 20
        if not pd.isna(row.get("rsi_slope")) and row.get("rsi_slope", 0) < 0 and row.get("rsi", 50) > 50: score += 20
        if not pd.isna(row.get("macd_hist")) and row.get("macd_hist", 0) < 0: score += 15
        if not pd.isna(row.get("dc_mid")) and row['close'] > row['dc_mid']: score += 15
        if not pd.isna(row.get("mfi")) and row.get("mfi", 50) < 70: score += 15
        if not pd.isna(row.get("atr")) and not pd.isna(row.get("atr_avg")) and row['atr'] > row['atr_avg']: score += 15

    # K > 70 bonus cho SHORT
    k_high = direction == "SHORT" and k > 70

    # ===== BBW FILTER (AFTER SCORING) =====
    # Kiểm tra BBW sau khi đã tính score xong
    # Nếu BBW < 0.002 → Return pending signal
    bbw = row.get('bbw', 0) if not pd.isna(row.get('bbw')) else 0
    if bbw < 0.002:
        # Signal detected but BBW too low → Return pending
        return direction, score, False, True  # (direction, score, is_reversal, is_pending)
    
    return direction, score, k_high, False  # (direction, score, is_reversal, is_pending)

# ==============================================================================
# TRADING LOGIC - TRADE MANAGEMENT
# ==============================================================================

def build_trade(row, direction):
    """
    Tính toán Entry, SL, TP cho một trade
    
    Logic:
    - LONG: SL = max(1%, ATR×1.5), TP = SL×2
    - SHORT: SL = 1%, TP = SL×2
    
    Returns:
        (entry, sl, tp)
    """
    entry = row['close']

    if direction == 'SHORT':
        sl_dist = entry * SL_SHORT_PCT
    else:
        atr_sl = row['atr_h1'] * SL_MULTIPLIER
        min_sl = entry * MIN_SL_PCT
        sl_dist = max(atr_sl, min_sl)

    if direction == 'LONG':
        sl = entry - sl_dist
        tp = entry + sl_dist * TP_MULTIPLIER
    else:
        sl = entry + sl_dist
        tp = entry - sl_dist * TP_MULTIPLIER

    return entry, sl, tp


class TradeState:
    """
    Quản lý trạng thái trade hiện tại
    """
    def __init__(self):
        self.position = None
        self.entry = None
        self.sl = None
        self.tp = None
        self.sl_moved = False

    def reset_position(self):
        """Reset tất cả trạng thái"""
        self.position = None
        self.entry = None
        self.sl = None
        self.tp = None
        self.sl_moved = False


state = TradeState()
pending = PendingSignal()  # Track pending signals when BBW is low


def check_exit(row):
    """
    Kiểm tra điều kiện exit (TP/SL/Breakeven)
    
    Returns:
        (result, price)
        - result: 'TP', 'SL', 'BE', hoặc None
        - price: Giá exit
    """
    high = row['high']
    low = row['low']
    close = row['close']
    sl_dist = abs(state.entry - state.sl) if state.sl else 0
    
    # Tính mức giá 1R (breakeven trigger)
    if state.position == "LONG":
        one_r_price = state.entry + sl_dist
    else:
        one_r_price = state.entry - sl_dist

    # ===== TRAILING STOP / BREAKEVEN LOGIC =====
    # Calculate trailing distance from ATR
    atr_h1 = row.get("atr_h1", 0) if not pd.isna(row.get("atr_h1")) else 0
    trailing_dist = atr_h1 * TRAILING_MULTIPLIER if atr_h1 > 0 else sl_dist * 0.5

    if state.position == "LONG":
        if high >= state.tp:
            return "TP", state.tp
        if close <= state.sl:
            return "SL", state.sl
        
        # Check Trailing / Breakeven
        if not state.sl_moved and high >= one_r_price:
            state.sl_moved = True
            if TRAILING_MULTIPLIER > 0:
                # Trailing mode: Start trailing
                new_sl = close - trailing_dist
                if new_sl > state.sl:
                    state.sl = new_sl
                    return "TRAILING", state.sl
            else:
                # Breakeven mode: Move SL to entry
                state.sl = state.entry
                return "BE", state.sl
        
        elif state.sl_moved and TRAILING_MULTIPLIER > 0:
             # Already trailing → Update SL if price moves up
            new_sl = close - trailing_dist
            if new_sl > state.sl:
                state.sl = new_sl
                return "TRAILING", state.sl

    if state.position == "SHORT":
        if low <= state.tp:
            return "TP", state.tp
        if close >= state.sl:
            return "SL", state.sl
            
        # Check Trailing / Breakeven
        if not state.sl_moved and low <= one_r_price:
            state.sl_moved = True
            if TRAILING_MULTIPLIER > 0:
                # Trailing mode: Start trailing
                new_sl = close + trailing_dist
                if new_sl < state.sl:
                    state.sl = new_sl
                    return "TRAILING", state.sl
            else:
                # Breakeven mode: Move SL to entry
                state.sl = state.entry
                return "BE", state.sl
        
        elif state.sl_moved and TRAILING_MULTIPLIER > 0:
            # Already trailing → Update SL if price moves down
            new_sl = close + trailing_dist
            if new_sl < state.sl:
                state.sl = new_sl
                return "TRAILING", state.sl

    return None, close


# ==============================================================================
# TELEGRAM NOTIFICATIONS
# ==============================================================================

async def send(bot, text):
    """Gửi message lên Telegram"""
    await bot.send_message(chat_id=CHANNEL_ID, text=text, parse_mode="HTML")


def format_signal(direction, entry, sl, tp, score, k_high=False):
    """Format message cho tín hiệu entry mới"""
    sl_pct = abs(entry - sl) / entry * 100
    tp_pct = abs(tp - entry) / entry * 100
    rr = tp_pct / sl_pct if sl_pct else 0
    
    score_display = f"{score}/100++" if k_high else f"{score}/100"

    return f"""
🤖 <b>{direction} BTCUSDT</b>

💰 Entry: <code>{entry:,.2f}</code>
🛑 SL: <code>{sl:,.2f}</code> ({sl_pct:.2f}%)
🎯 TP: <code>{tp:,.2f}</code> ({tp_pct:.2f}%)

📈 R:R = 1:{rr:.1f}
⭐ Score: {score_display}

⏰ {get_vn_time()}
"""


def format_close(result, price):
    """Format message cho đóng lệnh"""
    emoji = "🎯" if result == "TP" else "🛑"

    return f"""
{emoji} <b>{result} BTCUSDT</b>

📌 Vị thế: {state.position}
💰 Entry: <code>{state.entry:,.2f}</code>
🚪 Exit: <code>{price:,.2f}</code>

⏰ {get_vn_time()}
"""


    # Check logic for message formatting
    if "BE" in str(state.sl): # This logic is handled by caller, just keeping function generic
         pass
         
    return f"""
🔰 <b>BREAKEVEN BTCUSDT</b>

📌 Vị thế: {state.position}
💰 Entry: <code>{state.entry:,.2f}</code>
🛡️ SL mới: <code>{state.entry:,.2f}</code> (Entry)
🎯 TP: <code>{state.tp:,.2f}</code>

✅ Đã đạt 1R - Chuyển SL về Entry

⏰ {get_vn_time()}
"""

def format_trailing(new_sl):
    """Format message cho trailing stop update"""
    return f"""
🧗 <b>TRAILING STOP UPDATE</b>

📌 Vị thế: {state.position}
💰 Entry: <code>{state.entry:,.2f}</code>
🛡️ SL mới: <code>{new_sl:,.2f}</code>
🎯 TP: <code>{state.tp:,.2f}</code>

✅ Giá đang chạy tốt - Dời SL để lock profit

⏰ {get_vn_time()}
"""


# ==============================================================================
# TIMING & SCHEDULING
# ==============================================================================

def get_seconds_until_next_check():
    """
    Tính số giây đến mốc check tiếp theo (:00, :15, :30, :45) + buffer 30s
    """
    now = datetime.now(VN_TZ)
    current_minute = now.minute
    current_second = now.second
    
    # Tìm mốc tiếp theo (0, 15, 30, 45)
    next_minute_mark = ((current_minute // 15) + 1) * 15
    
    if next_minute_mark >= 60:
        minutes_to_wait = 60 - current_minute
    else:
        minutes_to_wait = next_minute_mark - current_minute
    
    # Tổng giây cần chờ + buffer 30s
    seconds_to_wait = (minutes_to_wait * 60) - current_second + 30
    
    next_check_time = now + timedelta(seconds=seconds_to_wait)
    logger.info(f"[TIMING] Next check at: {next_check_time.strftime('%H:%M:%S')}")
    
    return seconds_to_wait


# ==============================================================================
# MAIN LOOP
# ==============================================================================

async def run():
    """
    Main trading loop
    
    Flow:
    1. Gửi startup message
    2. Chờ đến mốc check tiếp theo
    3. Lấy data và tính indicators
    4. Phân tích tín hiệu
    5. Kiểm tra exit nếu đang có position
    6. Kiểm tra REV (reversal)
    7. Mở lệnh mới nếu có tín hiệu
    """
    bot = Bot(token=TELEGRAM_TOKEN)

    await send(bot, f"""
🚀 <b>BOT BTCUSDT KHỞI ĐỘNG</b>

⏱ Timeframe: M15 + H1
🎯 Chiến lược: Multi-indicator scoring
🔄 Cập nhật: Tối ưu logic REV và Hạn chế giao dịch khi Sideway

⏰ {get_vn_time()}
""")

    while True:
        # Chờ đến mốc check tiếp theo
        wait_seconds = get_seconds_until_next_check()
        logger.info(f"[TIMING] Waiting {wait_seconds} seconds until next check...")
        await asyncio.sleep(wait_seconds)
        
        try:
            # Lấy data và tính indicators
            df_m15, df_h1 = get_data()
            df = calculate_indicators(df_m15, df_h1)
            
            if df.empty:
                logger.warning("[WARNING] DataFrame trống, bỏ qua check này")
                continue
            
            last = df.iloc[-1]   # Nến đang chạy
            prev = df.iloc[-2] if len(df) > 1 else last  # Nến đã đóng
            current_price = last['close']
            
            logger.info(f"[CHECK] Time: {get_vn_time()}, Price: {current_price:.2f}")

            # Phân tích tín hiệu mới
            new_direction, score, k_high, is_pending = analyze_signal(last)

            if state.position:
                # Kiểm tra exit trên nến đã đóng
                result, price = check_exit(prev)
                
                # Kiểm tra REV (reversal)
                vwap = last.get('vwap', 0) if not pd.isna(last.get('vwap')) else 0
                vwap_ok = True
                if vwap > 0:
                    if state.position == "LONG" and new_direction == "SHORT" and current_price >= vwap:
                        vwap_ok = False
                    if state.position == "SHORT" and new_direction == "LONG" and current_price <= vwap:
                        vwap_ok = False
                
                min_rev = MIN_REV_LONG_SHORT if state.position == "LONG" else MIN_REV_SHORT_LONG
                if result is None and new_direction and new_direction != state.position and score >= min_rev and vwap_ok:
                    result = "REV"
                    price = current_price

                if result:
                    # Trailing / Breakeven: chỉ chuyển SL, không đóng lệnh
                    if result == "BE":
                        await send(bot, format_breakeven())
                    elif result == "TRAILING":
                         await send(bot, format_trailing(price))
                    else:
                        await send(bot, format_close(result, price))
                        state.reset_position()
                        
                        # Nếu là REV, mở lệnh mới ngay
                        if result == "REV" and new_direction and score >= MIN_REV_SCORE:
                            entry, sl, tp = build_trade(last, new_direction)
                            state.position = new_direction
                            state.entry, state.sl, state.tp = entry, sl, tp
                            await send(bot, format_signal(new_direction, entry, sl, tp, score, k_high))
            else:
                # ===== PENDING SIGNAL LOGIC =====
                bbw = last.get('bbw', 0) if not pd.isna(last.get('bbw')) else 0
                
                # Case 1: New pending signal (BBW < 0.002)
                if is_pending and new_direction:
                    pending.set_pending(new_direction, score, get_vn_time())
                    logger.info(f"[PENDING] {new_direction} signal pending (BBW too low: {bbw:.6f})")
                    continue
                
                # Case 2: BBW >= 0.002 and has pending signal
                if pending.has_pending() and bbw >= pending.bbw_threshold:
                    # Re-check signal direction
                    current_direction, current_score, current_k_high, _ = analyze_signal(last)
                    
                    if current_direction == pending.direction:
                        # Direction still valid → Execute pending signal!
                        new_direction = pending.direction
                        score = current_score  # Use current score
                        k_high = current_k_high
                        logger.info(f"[PENDING] Executing pending {new_direction} signal (BBW now: {bbw:.6f})")
                    else:
                        # Direction changed → Cancel pending
                        logger.info(f"[PENDING] Cancelled (direction changed from {pending.direction} to {current_direction})")
                        pending.clear()
                        continue
                    
                    pending.clear()
                
                # Không có position, kiểm tra entry mới
                min_score_required = MIN_SCORE_SHORT if new_direction == "SHORT" else MIN_SCORE
                if new_direction and score >= min_score_required:
                    entry, sl, tp = build_trade(last, new_direction)
                    state.position = new_direction
                    state.entry, state.sl, state.tp = entry, sl, tp
                    await send(bot, format_signal(new_direction, entry, sl, tp, score, k_high))
        
        except Exception as e:
            logger.error(f"[ERROR] Lỗi trong main loop: {e}")
            # Tiếp tục vòng lặp, không crash

if __name__ == "__main__":
    asyncio.run(run())