import json
import os
import sqlite3
from datetime import datetime, date, time as dtime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

load_dotenv()

# =============================================================
# AKÇAY RAILWAY BACKEND — EXECUTION READY v3.0
# -------------------------------------------------------------
# Supports:
# - TradingView TRADE_SIGNAL payloads from Gold/Silver Pine
# - TradingView PRICE_UPDATE payloads for runner/exit management
# - MT5 executor lifecycle sync: ACTIVE -> EXECUTED -> CLOSED
# - Backward-compatible legacy event="entry" payloads
# =============================================================

# =============================================================
# CONFIG
# =============================================================
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", os.getenv("TV_SECRET", "123abc456"))
DB = os.getenv("DB_NAME", os.getenv("DB_PATH", "akcay_mvp.db"))

# Risk amount sent to MT5 executor. Executor converts this into lots.
DEFAULT_RISK_GBP = float(os.getenv("DEFAULT_RISK_GBP", "200"))
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "20"))

# Freshness window for /trades?status=ACTIVE.
# Stale ACTIVE trades (Railway never received EXECUTED confirmation, but the
# real position was opened in MT5 yesterday/earlier) must NOT be re-served to
# the executor or it would re-open them after the local executed_trades.json
# rolls over at midnight.
MAX_ACTIVE_TRADE_AGE_SECONDS = int(os.getenv("MAX_ACTIVE_TRADE_AGE_SECONDS", "180"))

# Telegram optional
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "true").lower() == "true"

# Gate settings
SESSION_FILTER_ENABLED = os.getenv("SESSION_FILTER_ENABLED", "false").lower() == "true"
ENABLE_HTF_GATE = os.getenv("ENABLE_HTF_GATE", "false").lower() == "true"
ENABLE_CONFIDENCE_GATE = os.getenv("ENABLE_CONFIDENCE_GATE", "false").lower() == "true"
REJECT_LOW_CONFIDENCE = os.getenv("REJECT_LOW_CONFIDENCE", "false").lower() == "true"

MIN_CONFIDENCE_TO_ACCEPT = int(os.getenv("MIN_CONFIDENCE_TO_ACCEPT", "6"))
FULL_CONFIDENCE_THRESHOLD = int(os.getenv("FULL_CONFIDENCE_THRESHOLD", "9"))

app = FastAPI(title="AKÇAY Tactical Auto Trading — Execution Backend", version="3.0.0")

# =============================================================
# CONSTANTS
# =============================================================
HTF_STATE_MAP = {
    0: "NO_TRADE_ZONE",
    1: "LONG_ONLY",
    2: "SHORT_ONLY",
    3: "BOTH_ALLOWED",
}

SESSION_SCORE_MAP = {
    "BYPASSED": 0,
    "FULL": 2,
    "GOOD": 2,
    "TRADEABLE": 1,
    "WEAK": -2,
    "REDUCED": -2,
    "NEUTRAL": 0,
    "UNKNOWN": 0,
    "REJECT": -99,
}

VALID_INSTRUMENTS = {"GOLD", "SILVER"}
VALID_DIRECTIONS = {"BUY", "SELL", "LONG", "SHORT"}

# =============================================================
# MODELS
# =============================================================
class RunnerConfig(BaseModel):
    tp1_close_pct: Optional[int] = 30
    tp2_close_pct: Optional[int] = 30
    runner_pct: Optional[int] = 40
    be_offset_after_tp2: Optional[float] = 0.1
    trail_pivot_left: Optional[int] = 2
    trail_pivot_right: Optional[int] = 2
    trail_buffer_atr_mult: Optional[float] = 0.2
    trail_buffer_min: Optional[float] = 0.5
    trail_buffer_max: Optional[float] = 8.0
    time_stop_bars_aplus: Optional[int] = 25
    time_stop_bars_a: Optional[int] = 15
    time_stop_progress_window: Optional[int] = 1
    exit_requires_ema_break: Optional[bool] = True
    exit_requires_swing_break: Optional[bool] = True

    class Config:
        extra = "allow"


class TradeSignalPayload(BaseModel):
    secret: Optional[str] = None
    event: Optional[str] = "TRADE_SIGNAL"
    bar_time: Optional[int] = None
    instrument: str
    direction: str
    entry: float
    stop: float
    tp: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    current_price: Optional[float] = None
    setup: Optional[str] = "default"
    quality: Optional[str] = "A"
    regime_score: Optional[int] = 0
    atr_expanding: Optional[Any] = False
    risk_per_unit: Optional[float] = None
    atr_value: Optional[float] = None
    runner_config: Optional[RunnerConfig] = Field(default_factory=RunnerConfig)
    trend_state: Optional[Dict[str, Any]] = None

    # Legacy HTF fields from older Silver script
    htf_tf: Optional[str] = "120"
    htf_state_code: Optional[int] = 3
    htf_score: Optional[int] = 0

    class Config:
        extra = "allow"


class PriceUpdatePayload(BaseModel):
    secret: Optional[str] = None
    event: Optional[str] = "PRICE_UPDATE"
    bar_time: Optional[int] = None
    instrument: str
    current_price: float
    ema20: Optional[float] = None
    atr_value: Optional[float] = None
    rsi: Optional[float] = None
    atr_expanding: Optional[Any] = False
    ema_slope: Optional[float] = None
    regime_score: Optional[int] = 0
    current_15m_bias: Optional[str] = None
    latest_swing_low: Optional[float] = None
    latest_swing_high: Optional[float] = None
    trail_ref_low: Optional[float] = None
    trail_ref_high: Optional[float] = None

    class Config:
        extra = "allow"


class TradeStatusUpdate(BaseModel):
    status: str
    reason: Optional[str] = None
    close_reason: Optional[str] = None
    mt5_order: Optional[Any] = None
    mt5_deal: Optional[Any] = None

    class Config:
        extra = "allow"


# Legacy models kept for compatibility only.
class DailyBiasPayload(BaseModel):
    secret: str
    system: Literal["AKCAY_DAILY_BIAS"]
    ticker: str
    bias: Literal["LONG", "SHORT", "NEUTRAL"]
    price: float
    tf: str


class SignalPayload(BaseModel):
    secret: str
    system: Literal["PLAN_A_EM"]
    ticker: str
    side: Literal["LONG", "SHORT", "ANY"]
    price: float
    tf: str
    action: str = "REVIEW"

# =============================================================
# DATABASE
# =============================================================
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = db()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_uid TEXT UNIQUE,
        instrument TEXT NOT NULL,
        direction TEXT NOT NULL,
        entry REAL NOT NULL,
        stop REAL NOT NULL,
        tp REAL,
        tp1 REAL,
        tp2 REAL,
        current_price REAL,
        setup TEXT,
        quality TEXT,
        regime_score INTEGER,
        atr_expanding INTEGER,
        htf_tf TEXT,
        htf_state_code INTEGER,
        htf_state TEXT,
        htf_score INTEGER,
        session_quality TEXT,
        confidence_score INTEGER,
        confidence_mode TEXT,
        risk_amount_gbp REAL,
        risk_per_unit REAL,
        atr_value REAL,
        runner_config TEXT,
        trend_state TEXT,
        raw_payload TEXT,
        status TEXT NOT NULL,
        reason TEXT,
        bar_time INTEGER,
        created_at TEXT NOT NULL,
        updated_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_uid TEXT,
        instrument TEXT,
        direction TEXT,
        setup TEXT,
        decision TEXT,
        reason TEXT,
        regime_score INTEGER,
        htf_state_code INTEGER,
        htf_state TEXT,
        confidence_score INTEGER,
        confidence_mode TEXT,
        entry REAL,
        stop REAL,
        tp REAL,
        tp1 REAL,
        tp2 REAL,
        raw_payload TEXT,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS market_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instrument TEXT UNIQUE,
        current_price REAL,
        ema20 REAL,
        atr_value REAL,
        rsi REAL,
        atr_expanding INTEGER,
        ema_slope REAL,
        regime_score INTEGER,
        current_15m_bias TEXT,
        latest_swing_low REAL,
        latest_swing_high REAL,
        trail_ref_low REAL,
        trail_ref_high REAL,
        raw_payload TEXT,
        bar_time INTEGER,
        updated_at TEXT NOT NULL
    )
    """)

    # Migration: add trail_ref columns to existing market_state tables.
    # CREATE TABLE IF NOT EXISTS is a no-op when the table already exists,
    # so older deployments need an explicit ALTER. Both columns wrapped in
    # try/except because SQLite raises if the column already exists, and
    # there's no portable IF NOT EXISTS for ALTER TABLE ADD COLUMN.
    for col_name in ("trail_ref_low", "trail_ref_high"):
        try:
            cur.execute(f"ALTER TABLE market_state ADD COLUMN {col_name} REAL")
            print(f"MIGRATION: added column market_state.{col_name}")
        except Exception:
            # Column already exists — safe to ignore.
            pass

    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily_bias (
        ticker TEXT PRIMARY KEY,
        bias TEXT NOT NULL,
        valid_date TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    con.commit()
    con.close()


def migrate_db():
    con = db()
    cur = con.cursor()

    def add_column_if_missing(table: str, column: str, definition: str):
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    for col, definition in {
        "tp1": "REAL",
        "tp2": "REAL",
        "risk_per_unit": "REAL",
        "atr_value": "REAL",
        "runner_config": "TEXT",
        "trend_state": "TEXT",
        "raw_payload": "TEXT",
        "bar_time": "INTEGER",
        "updated_at": "TEXT",
    }.items():
        add_column_if_missing("trades", col, definition)

    for col, definition in {
        "tp1": "REAL",
        "tp2": "REAL",
        "raw_payload": "TEXT",
    }.items():
        add_column_if_missing("signals", col, definition)

    con.commit()
    con.close()


init_db()
migrate_db()

# =============================================================
# HELPERS
# =============================================================
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_str() -> str:
    return str(date.today())


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bool_to_int(value: Any) -> int:
    return 1 if str(value).lower() in {"true", "1", "yes"} else 0


def normalize_direction(direction: str) -> str:
    d = str(direction or "").upper().strip()
    if d in {"BUY", "LONG"}:
        return "BUY"
    if d in {"SELL", "SHORT"}:
        return "SELL"
    raise HTTPException(status_code=400, detail="Invalid direction")


def normalize_instrument(instrument: str) -> str:
    i = str(instrument or "").upper().strip()
    if i not in VALID_INSTRUMENTS:
        raise HTTPException(status_code=400, detail="Invalid instrument")
    return i


def check_secret(secret: Optional[str]):
    if WEBHOOK_SECRET and secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")


def htf_state_text(code: int) -> str:
    return HTF_STATE_MAP.get(code, "UNKNOWN")


def get_htf_fields(payload: TradeSignalPayload) -> Tuple[int, str, int, str]:
    trend_state = payload.trend_state or {}
    htf_code = safe_int(trend_state.get("htf_state_code", payload.htf_state_code), 3)
    htf_score = safe_int(trend_state.get("htf_score", payload.htf_score), 0)
    current_15m_bias = str(trend_state.get("current_15m_bias", ""))
    return htf_code, htf_state_text(htf_code), htf_score, current_15m_bias


def make_trade_uid(payload: TradeSignalPayload) -> str:
    # Pine sends bar_time; use it to avoid duplicate micro differences while allowing future signals.
    entry = round(float(payload.entry), 3)
    stop = round(float(payload.stop), 3)
    tp2 = round(float(payload.tp2 if payload.tp2 is not None else payload.tp or 0), 3)
    bar_time = payload.bar_time or int(datetime.now(timezone.utc).timestamp() // 60)
    direction = normalize_direction(payload.direction)
    instrument = normalize_instrument(payload.instrument)
    return f"{instrument}:{direction}:{payload.setup}:{entry}:{stop}:{tp2}:{bar_time}"


def raw_json(model: BaseModel) -> str:
    return model.model_dump_json()

# =============================================================
# LOGGING / STORAGE
# =============================================================
def log_signal(payload: TradeSignalPayload, trade_uid: str, decision: str, reason: str, htf_code: int, htf_state: str, score: Optional[int], mode: Optional[str]):
    con = db()
    cur = con.cursor()
    tp2_or_tp = payload.tp2 if payload.tp2 is not None else payload.tp
    cur.execute("""
        INSERT INTO signals
        (trade_uid, instrument, direction, setup, decision, reason,
         regime_score, htf_state_code, htf_state, confidence_score,
         confidence_mode, entry, stop, tp, tp1, tp2, raw_payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_uid,
        normalize_instrument(payload.instrument),
        normalize_direction(payload.direction),
        payload.setup,
        decision,
        reason,
        safe_int(payload.regime_score, 0),
        htf_code,
        htf_state,
        score,
        mode,
        safe_float(payload.entry),
        safe_float(payload.stop),
        safe_float(tp2_or_tp),
        safe_float(payload.tp1),
        safe_float(tp2_or_tp),
        raw_json(payload),
        now(),
    ))
    con.commit()
    con.close()


def save_trade(payload: TradeSignalPayload, trade_uid: str, status: str, reason: str, htf_state: str, htf_code: int, htf_score: int, session_quality: str, confidence_score: int, confidence_mode: str, risk_amount_gbp: float):
    con = db()
    cur = con.cursor()
    tp2_or_tp = payload.tp2 if payload.tp2 is not None else payload.tp
    risk_per_unit = payload.risk_per_unit if payload.risk_per_unit is not None else abs(payload.entry - payload.stop)
    cur.execute("""
        INSERT OR IGNORE INTO trades
        (trade_uid, instrument, direction, entry, stop, tp, tp1, tp2, current_price,
         setup, quality, regime_score, atr_expanding, htf_tf, htf_state_code,
         htf_state, htf_score, session_quality, confidence_score,
         confidence_mode, risk_amount_gbp, risk_per_unit, atr_value,
         runner_config, trend_state, raw_payload, status, reason, bar_time, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_uid,
        normalize_instrument(payload.instrument),
        normalize_direction(payload.direction),
        safe_float(payload.entry),
        safe_float(payload.stop),
        safe_float(tp2_or_tp),
        safe_float(payload.tp1),
        safe_float(tp2_or_tp),
        safe_float(payload.current_price if payload.current_price is not None else payload.entry),
        payload.setup,
        payload.quality,
        safe_int(payload.regime_score, 0),
        bool_to_int(payload.atr_expanding),
        str(payload.htf_tf or "120"),
        htf_code,
        htf_state,
        htf_score,
        session_quality,
        confidence_score,
        confidence_mode,
        risk_amount_gbp,
        safe_float(risk_per_unit),
        safe_float(payload.atr_value),
        payload.runner_config.model_dump_json() if payload.runner_config else "{}",
        json.dumps(payload.trend_state or {}, ensure_ascii=False),
        raw_json(payload),
        status,
        reason,
        safe_int(payload.bar_time, 0),
        now(),
        now(),
    ))
    con.commit()
    con.close()


def trade_uid_exists(trade_uid: str) -> bool:
    con = db()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM trades WHERE trade_uid = ?", (trade_uid,))
    count = cur.fetchone()[0]
    con.close()
    return count > 0


def active_trade_count() -> int:
    con = db()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM trades WHERE status = 'ACTIVE'")
    count = cur.fetchone()[0]
    con.close()
    return count


def daily_trade_count() -> int:
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM trades
        WHERE status IN ('ACTIVE','EXECUTED') AND substr(created_at, 1, 10) = ?
    """, (today_str(),))
    count = cur.fetchone()[0]
    con.close()
    return count


# =============================================================
# SESSION / HTF / CONFIDENCE
# =============================================================
def current_utc_time() -> dtime:
    return datetime.utcnow().time()


def in_time_range(t: dtime, start: dtime, end: dtime) -> bool:
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end


def classify_session(instrument: str) -> Tuple[str, str]:
    if not SESSION_FILTER_ENABLED:
        return "BYPASSED", "SESSION_FILTER_DISABLED"

    t = current_utc_time()
    instrument = instrument.upper()

    if in_time_range(t, dtime(21, 55), dtime(22, 10)):
        return "REJECT", "DEAD_ROLLOVER_REJECT"

    if instrument == "GOLD":
        if in_time_range(t, dtime(7, 0), dtime(17, 30)):
            return "FULL", "GOLD_LONDON_NY_FULL"
        if in_time_range(t, dtime(23, 0), dtime(6, 59)):
            return "WEAK", "GOLD_ASIA_WEAK"
        return "TRADEABLE", "GOLD_OTHER_TRADEABLE"

    if instrument == "SILVER":
        if in_time_range(t, dtime(7, 0), dtime(16, 30)):
            return "FULL", "SILVER_LONDON_NY_FULL"
        if in_time_range(t, dtime(17, 0), dtime(20, 59)):
            return "WEAK", "SILVER_LATE_WEAK"
        return "REJECT", "SILVER_OUTSIDE_ALLOWED_REJECT"

    return "REJECT", "INVALID_INSTRUMENT_SESSION_REJECT"


def htf_bias_allows_trade(direction: str, htf_state_code: int) -> Tuple[bool, str]:
    if not ENABLE_HTF_GATE:
        return True, "HTF_GATE_DISABLED"
    if htf_state_code == 0:
        return False, "HTF_NO_TRADE_ZONE_REJECT"
    if direction == "BUY" and htf_state_code == 2:
        return False, "HTF_SHORT_ONLY_REJECT"
    if direction == "SELL" and htf_state_code == 1:
        return False, "HTF_LONG_ONLY_REJECT"
    return True, "HTF_OK"


def calculate_confidence(payload: TradeSignalPayload, session_quality: str, htf_code: int, htf_score: int) -> Tuple[int, List[str]]:
    regime_score = safe_int(payload.regime_score, 0)
    atr_expanding = bool_to_int(payload.atr_expanding)
    quality = str(payload.quality or "A").upper().strip()
    setup = str(payload.setup or "").upper().strip()

    score = 0
    reasons: List[str] = []

    if htf_code in {1, 2}:
        score += 3
        reasons.append(f"HTF_STRONG:{htf_state_text(htf_code)}")
    elif htf_code == 3:
        score += 1
        reasons.append("HTF_BOTH_ALLOWED")
    else:
        reasons.append("HTF_NO_TRADE_ZONE")

    if htf_score >= 5:
        score += 1
        reasons.append("HTF_SCORE_5")

    if regime_score >= 5:
        score += 3
        reasons.append(f"REGIME_STRONG:R{regime_score}")
    elif regime_score >= 3:
        score += 1
        reasons.append(f"REGIME_TRADEABLE:R{regime_score}")
    else:
        score -= 3
        reasons.append(f"REGIME_WEAK:R{regime_score}")

    if atr_expanding == 1:
        score += 1
        reasons.append("ATR_EXPANDING")

    if quality == "A+":
        score += 1
        reasons.append("QUALITY_A_PLUS")

    if setup in {"BO_LONG", "BO_SHORT", "BREAKOUT_CONTINUATION_LONG", "BREAKDOWN_CONTINUATION_SHORT"}:
        score += 1
        reasons.append("BO_OR_BREAKOUT_SETUP")

    session_score = SESSION_SCORE_MAP.get(session_quality, 0)
    score += session_score
    reasons.append(f"SESSION_{session_quality}:{session_score}")

    return score, reasons


def confidence_mode(score: int) -> str:
    if not ENABLE_CONFIDENCE_GATE:
        return "LOG_ONLY"
    if score < MIN_CONFIDENCE_TO_ACCEPT:
        return "NO_EXECUTE"
    if score < FULL_CONFIDENCE_THRESHOLD:
        return "REDUCED"
    return "FULL"

# =============================================================
# TELEGRAM
# =============================================================
def send_telegram(text: str):
    if not ENABLE_TELEGRAM:
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=8)
    except Exception as exc:
        print(f"Telegram error: {exc}")


def telegram_trade_message(payload: TradeSignalPayload, decision: str, reason: str, htf_state: str, session_quality: str, score: int, mode: str) -> str:
    icon = "✅" if decision == "ACTIVE" else "❌"
    tp2_or_tp = payload.tp2 if payload.tp2 is not None else payload.tp
    return (
        f"{icon} {decision} — {normalize_instrument(payload.instrument)} {normalize_direction(payload.direction)}\n"
        f"Setup: {payload.setup} | Quality: {payload.quality}\n"
        f"Entry: {payload.entry} | SL: {payload.stop}\n"
        f"TP1: {payload.tp1} | TP2: {tp2_or_tp}\n"
        f"Regime: R{payload.regime_score} | ATR expanding: {payload.atr_expanding}\n"
        f"HTF: {htf_state} | Session: {session_quality}\n"
        f"Confidence: {score} / Mode: {mode}\n"
        f"Risk: £{DEFAULT_RISK_GBP}\n"
        f"Reason: {reason}\n"
        f"Time: {now()}"
    )

# =============================================================
# WEBHOOK ROUTER
# =============================================================
def normalize_raw_body(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    raise HTTPException(status_code=422, detail="invalid JSON body")


async def route_webhook(request: Request) -> Dict[str, Any]:
    try:
        raw = await request.json()
    except Exception:
        body = (await request.body()).decode("utf-8", errors="ignore").strip()
        raw = body

    payload = normalize_raw_body(raw)
    event = str(payload.get("event", "entry")).upper().strip()

    if event in {"TRADE_SIGNAL", "ENTRY"}:
        return handle_trade_signal(TradeSignalPayload(**payload))

    if event == "PRICE_UPDATE":
        return handle_price_update(PriceUpdatePayload(**payload))

    raise HTTPException(status_code=422, detail=f"unsupported event: {event}")


def handle_price_update(payload: PriceUpdatePayload) -> Dict[str, Any]:
    check_secret(payload.secret)
    instrument = normalize_instrument(payload.instrument)

    con = db()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO market_state
        (instrument, current_price, ema20, atr_value, rsi, atr_expanding, ema_slope,
         regime_score, current_15m_bias, latest_swing_low, latest_swing_high,
         trail_ref_low, trail_ref_high,
         raw_payload, bar_time, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(instrument) DO UPDATE SET
            current_price=excluded.current_price,
            ema20=excluded.ema20,
            atr_value=excluded.atr_value,
            rsi=excluded.rsi,
            atr_expanding=excluded.atr_expanding,
            ema_slope=excluded.ema_slope,
            regime_score=excluded.regime_score,
            current_15m_bias=excluded.current_15m_bias,
            latest_swing_low=excluded.latest_swing_low,
            latest_swing_high=excluded.latest_swing_high,
            trail_ref_low=excluded.trail_ref_low,
            trail_ref_high=excluded.trail_ref_high,
            raw_payload=excluded.raw_payload,
            bar_time=excluded.bar_time,
            updated_at=excluded.updated_at
    """, (
        instrument,
        safe_float(payload.current_price),
        safe_float(payload.ema20),
        safe_float(payload.atr_value),
        safe_float(payload.rsi),
        bool_to_int(payload.atr_expanding),
        safe_float(payload.ema_slope),
        safe_int(payload.regime_score, 0),
        payload.current_15m_bias,
        safe_float(payload.latest_swing_low),
        safe_float(payload.latest_swing_high),
        safe_float(payload.trail_ref_low),
        safe_float(payload.trail_ref_high),
        raw_json(payload),
        safe_int(payload.bar_time, 0),
        now(),
    ))
    con.commit()
    con.close()
    return {"status": "PRICE_UPDATE_OK", "instrument": instrument, "bar_time": payload.bar_time}


def handle_trade_signal(payload: TradeSignalPayload) -> Dict[str, Any]:
    check_secret(payload.secret)

    instrument = normalize_instrument(payload.instrument)
    direction = normalize_direction(payload.direction)
    tp2_or_tp = payload.tp2 if payload.tp2 is not None else payload.tp

    if payload.entry <= 0 or payload.stop <= 0:
        raise HTTPException(status_code=400, detail="entry/stop must be > 0")
    if tp2_or_tp is None or safe_float(tp2_or_tp) <= 0:
        raise HTTPException(status_code=400, detail="tp2 or tp must be > 0")
    if payload.tp1 is None:
        # Legacy fallback: TP1 halfway between entry and final TP.
        payload.tp1 = payload.entry + ((safe_float(tp2_or_tp) - payload.entry) * 0.5)
    if payload.current_price is None:
        payload.current_price = payload.entry
    if payload.risk_per_unit is None:
        payload.risk_per_unit = abs(payload.entry - payload.stop)

    htf_code, htf_state, htf_score, _ = get_htf_fields(payload)
    trade_uid = make_trade_uid(payload)

    if trade_uid_exists(trade_uid):
        reason = "DUPLICATE_SIGNAL_REJECT"
        log_signal(payload, trade_uid, "REJECTED", reason, htf_code, htf_state, None, None)
        send_telegram(telegram_trade_message(payload, "REJECTED", reason, htf_state, "UNKNOWN", 0, "NO_EXECUTE"))
        return {"decision": "REJECTED", "reason": reason, "trade_uid": trade_uid}

    session_quality, session_reason = classify_session(instrument)
    if session_quality == "REJECT":
        reason = session_reason
        score, _ = calculate_confidence(payload, "UNKNOWN", htf_code, htf_score)
        mode = "NO_EXECUTE"
        save_trade(payload, trade_uid, "REJECTED", reason, htf_state, htf_code, htf_score, session_quality, score, mode, 0.0)
        log_signal(payload, trade_uid, "REJECTED", reason, htf_code, htf_state, score, mode)
        # Telegram notification intentionally suppressed for session rejects.
        # DB + log preserved for audit; user requested quieter rejection chatter.
        return {"decision": "REJECTED", "reason": reason, "trade_uid": trade_uid}

    htf_allowed, htf_reason = htf_bias_allows_trade(direction, htf_code)
    if not htf_allowed:
        reason = htf_reason
        score, _ = calculate_confidence(payload, session_quality, htf_code, htf_score)
        mode = "NO_EXECUTE"
        save_trade(payload, trade_uid, "REJECTED", reason, htf_state, htf_code, htf_score, session_quality, score, mode, 0.0)
        log_signal(payload, trade_uid, "REJECTED", reason, htf_code, htf_state, score, mode)
        # Telegram notification intentionally suppressed for HTF rejects.
        return {"decision": "REJECTED", "reason": reason, "trade_uid": trade_uid, "htf_state": htf_state}

    if daily_trade_count() >= MAX_DAILY_TRADES:
        reason = "DAILY_TRADE_LIMIT_REJECT"
        score, _ = calculate_confidence(payload, session_quality, htf_code, htf_score)
        mode = "NO_EXECUTE"
        save_trade(payload, trade_uid, "REJECTED", reason, htf_state, htf_code, htf_score, session_quality, score, mode, 0.0)
        log_signal(payload, trade_uid, "REJECTED", reason, htf_code, htf_state, score, mode)
        # Telegram notification intentionally suppressed for trade-count rejects.
        return {"decision": "REJECTED", "reason": reason, "trade_uid": trade_uid}

    # Daily risk gate intentionally REMOVED from Railway.
    # All risk-based execution decisions (soft/hard tiered limits, half-risk
    # fallback, lot scaling, daily budget tracking) now live exclusively in
    # the MT5 executor, which has access to live account balance and live
    # position state. Railway's job is to validate the payload, deduplicate,
    # store the signal, and forward it to the executor as ACTIVE.

    score, score_reasons = calculate_confidence(payload, session_quality, htf_code, htf_score)
    mode = confidence_mode(score)

    if REJECT_LOW_CONFIDENCE and mode == "NO_EXECUTE":
        reason = "LOW_CONFIDENCE_REJECT"
        save_trade(payload, trade_uid, "REJECTED", reason, htf_state, htf_code, htf_score, session_quality, score, mode, 0.0)
        log_signal(payload, trade_uid, "REJECTED", reason, htf_code, htf_state, score, mode)
        send_telegram(telegram_trade_message(payload, "REJECTED", reason, htf_state, session_quality, score, mode))
        return {"decision": "REJECTED", "reason": reason, "trade_uid": trade_uid, "confidence_score": score, "confidence_reasons": score_reasons}

    reason = "ALL_FILTERS_PASSED"
    save_trade(payload, trade_uid, "ACTIVE", reason, htf_state, htf_code, htf_score, session_quality, score, mode, DEFAULT_RISK_GBP)
    log_signal(payload, trade_uid, "ACTIVE", reason, htf_code, htf_state, score, mode)
    send_telegram(telegram_trade_message(payload, "ACTIVE", reason, htf_state, session_quality, score, mode))

    return {
        "decision": "ACTIVE",
        "reason": reason,
        "trade_uid": trade_uid,
        "htf_state": htf_state,
        "session_quality": session_quality,
        "confidence_score": score,
        "confidence_mode": mode,
        "confidence_reasons": score_reasons,
    }

# =============================================================
# ROUTES
# =============================================================
@app.post("/webhook/tradingview")
async def webhook_tradingview(request: Request):
    return await route_webhook(request)


@app.post("/webhook/entry")
async def webhook_entry(request: Request):
    return await route_webhook(request)


@app.get("/trades")
def get_trades(status: Optional[str] = Query(default="ACTIVE"), limit: int = 100):
    con = db()
    cur = con.cursor()

    if status is None or status == "":
        cur.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
    elif str(status).upper() == "ACTIVE":
        # Freshness guard: only return ACTIVE trades that are still recent.
        # Prevents the executor from re-opening stale ACTIVE trades the day
        # after a missed status update (MT5 position was actually opened, but
        # Railway never received the EXECUTED confirmation). Also blocks
        # signals that the executor wasn't running to consume in time.
        now_unix = int(datetime.now(timezone.utc).timestamp())
        unix_cutoff = now_unix - MAX_ACTIVE_TRADE_AGE_SECONDS

        iso_cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=MAX_ACTIVE_TRADE_AGE_SECONDS)
        ).isoformat()

        cur.execute(
            """
            SELECT * FROM trades
            WHERE status = 'ACTIVE'
              AND (
                    (bar_time IS NOT NULL AND bar_time > 0 AND bar_time >= ?)
                 OR created_at >= ?
              )
            ORDER BY id DESC
            LIMIT ?
            """,
            (unix_cutoff, iso_cutoff, limit),
        )
    else:
        cur.execute("SELECT * FROM trades WHERE status = ? ORDER BY id DESC LIMIT ?", (status, limit))

    rows = [dict(row) for row in cur.fetchall()]
    con.close()
    return rows


@app.post("/trades/{trade_id}/status")
def update_trade_status(trade_id: int, payload: TradeStatusUpdate):
    new_status = str(payload.status or "").upper().strip()
    if new_status not in {"ACTIVE", "EXECUTED", "CLOSED", "REJECTED", "CANCELLED"}:
        raise HTTPException(status_code=422, detail=f"invalid status: {payload.status}")

    reason = payload.reason or payload.close_reason or "STATUS_UPDATED"
    con = db()
    cur = con.cursor()
    cur.execute("SELECT id FROM trades WHERE id = ?", (trade_id,))
    row = cur.fetchone()
    if row is None:
        con.close()
        raise HTTPException(status_code=404, detail="trade not found")

    cur.execute("""
        UPDATE trades
        SET status = ?, reason = ?, updated_at = ?
        WHERE id = ?
    """, (new_status, reason, now(), trade_id))
    con.commit()
    con.close()

    return {"status": "ok", "trade_id": trade_id, "new_status": new_status, "reason": reason}


@app.get("/market-state")
def get_market_state(instrument: Optional[str] = None):
    con = db()
    cur = con.cursor()
    if instrument:
        cur.execute("SELECT * FROM market_state WHERE instrument = ?", (instrument.upper(),))
        row = cur.fetchone()
        con.close()
        return dict(row) if row else {}

    cur.execute("SELECT * FROM market_state ORDER BY updated_at DESC")
    rows = [dict(row) for row in cur.fetchall()]
    con.close()
    return rows


@app.get("/signals")
def get_signals(limit: int = 100):
    con = db()
    cur = con.cursor()
    cur.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    con.close()
    return rows


@app.get("/health")
def health():
    return {
        "status": "running",
        "time": now(),
        "version": "execution_ready_v3",
        "supported_events": ["TRADE_SIGNAL", "PRICE_UPDATE", "entry"],
        "session_filter_enabled": SESSION_FILTER_ENABLED,
        "htf_gate": ENABLE_HTF_GATE,
        "confidence_gate": ENABLE_CONFIDENCE_GATE,
        "reject_low_confidence": REJECT_LOW_CONFIDENCE,
        "default_risk_gbp": DEFAULT_RISK_GBP,
        "max_daily_trades": MAX_DAILY_TRADES,
    }


@app.get("/status")
def status():
    return health()

# =============================================================
# LEGACY ROUTES — KEPT FOR COMPATIBILITY, NO DIRECT MT5 EXECUTION
# =============================================================
def save_daily_bias(ticker: str, bias: str):
    con = db()
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO daily_bias
        (ticker, bias, valid_date, created_at)
        VALUES (?, ?, ?, ?)
    """, (ticker, bias, today_str(), now()))
    con.commit()
    con.close()


@app.post("/webhook/daily-bias")
def daily_bias(payload: DailyBiasPayload):
    check_secret(payload.secret)
    save_daily_bias(payload.ticker, payload.bias)
    return {"status": "ok", "ticker": payload.ticker, "bias": payload.bias, "message": "Daily bias saved"}


@app.post("/webhook/plan-a")
def plan_a_signal(payload: SignalPayload):
    check_secret(payload.secret)
    return {"decision": "IGNORE", "reason": "LEGACY_PLAN_A_DISABLED_USE_WEBHOOK_ENTRY", "ticker": payload.ticker, "side": payload.side}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
