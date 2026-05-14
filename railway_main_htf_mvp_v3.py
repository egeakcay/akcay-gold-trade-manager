# railway_main_gold_event_router_v1.py

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Any, Dict, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

# =============================================================
# ENV / CONFIG
# =============================================================
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
DB_PATH = os.getenv("DB_PATH", "trades.db")

ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "true").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SESSION_FILTER_ENABLED = os.getenv("SESSION_FILTER_ENABLED", "false").lower() == "true"
ENABLE_HTF_GATE = os.getenv("ENABLE_HTF_GATE", "false").lower() == "true"
ENABLE_CONFIDENCE_GATE = os.getenv("ENABLE_CONFIDENCE_GATE", "false").lower() == "true"
REJECT_LOW_CONFIDENCE = os.getenv("REJECT_LOW_CONFIDENCE", "false").lower() == "true"

CONFIDENCE_MIN = int(os.getenv("CONFIDENCE_MIN", "60"))
DAILY_TRADE_LIMIT = int(os.getenv("DAILY_TRADE_LIMIT", "10"))

ACCOUNT_BALANCE_GBP = float(os.getenv("ACCOUNT_BALANCE_GBP", "10000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "2"))

DUPLICATE_WINDOW_MINUTES = int(os.getenv("DUPLICATE_WINDOW_MINUTES", "15"))

HTF_STATE_MAP = {
    0: "BEARISH_ONLY",
    1: "BULLISH_ONLY",
    2: "NEUTRAL_BLOCKED",
    3: "BOTH_ALLOWED",
}

GOLD_SYMBOLS = {"XAUUSD", "GOLD", "XAU/USD", "XAU_USD", "XAUUSDM", "XAUUSDC"}
SILVER_SYMBOLS = {"XAGUSD", "SILVER", "XAG/USD", "XAG_USD", "XAGUSDM", "XAGUSDC"}

# =============================================================
# APP
# =============================================================
app = FastAPI(title="AKÇAY Event Router", version="1.0.0")

# =============================================================
# UTILS
# =============================================================
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value or {}, ensure_ascii=False)
    except Exception:
        return "{}"


def make_trade_uid(instrument: str, direction: str, entry: float, setup: str, bar_time: Optional[int] = None) -> str:
    minute_key = bar_time or int(datetime.now(timezone.utc).timestamp() // 60)
    raw = f"{instrument}|{direction}|{round(entry, 5)}|{setup}|{minute_key}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def classify_instrument(instrument: str) -> str:
    if not instrument:
        return "OTHER"
    sym = instrument.upper().replace(" ", "")
    if sym in GOLD_SYMBOLS or "XAUUSD" in sym or sym.startswith("XAU"):
        return "GOLD"
    if sym in SILVER_SYMBOLS or "XAGUSD" in sym or sym.startswith("XAG"):
        return "SILVER"
    return "OTHER"


def normalize_direction(direction: str) -> str:
    d = str(direction or "").upper().strip()
    if d in {"LONG", "BUY"}:
        return "BUY"
    if d in {"SHORT", "SELL"}:
        return "SELL"
    raise ValueError(f"invalid direction: {direction}")

# =============================================================
# DB
# =============================================================
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = db()
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT,
            trade_uid TEXT,
            instrument TEXT,
            direction TEXT,
            setup TEXT,
            quality TEXT,
            decision TEXT,
            reason TEXT,
            regime_score INTEGER,
            atr_expanding INTEGER,
            entry REAL,
            stop REAL,
            tp REAL,
            tp1 REAL,
            tp2 REAL,
            current_price REAL,
            risk_per_unit REAL,
            atr_value REAL,
            risk_amount_gbp REAL,
            runner_config TEXT,
            trend_state TEXT,
            raw_payload TEXT,
            telegram_sent INTEGER DEFAULT 0,
            bar_time INTEGER,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_uid TEXT UNIQUE,
            instrument TEXT,
            direction TEXT,
            entry REAL,
            stop REAL,
            tp REAL,
            tp1 REAL,
            tp2 REAL,
            current_price REAL,
            setup TEXT,
            quality TEXT,
            regime_score INTEGER,
            atr_expanding INTEGER,
            raw_payload TEXT,
            risk_amount_gbp REAL,
            risk_per_unit REAL,
            atr_value REAL,
            runner_config TEXT,
            trend_state TEXT,
            status TEXT,
            reason TEXT,
            bar_time INTEGER,
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
            raw_payload TEXT,
            bar_time INTEGER,
            updated_at TEXT NOT NULL
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

    for table in ["signals", "trades"]:
        add_column_if_missing(table, "event", "TEXT")
        add_column_if_missing(table, "tp1", "REAL")
        add_column_if_missing(table, "tp2", "REAL")
        add_column_if_missing(table, "risk_per_unit", "REAL")
        add_column_if_missing(table, "atr_value", "REAL")
        add_column_if_missing(table, "runner_config", "TEXT")
        add_column_if_missing(table, "trend_state", "TEXT")
        add_column_if_missing(table, "bar_time", "INTEGER")

    con.commit()
    con.close()

# =============================================================
# PAYLOADS
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
    event: str = "TRADE_SIGNAL"
    bar_time: Optional[int] = None
    instrument: str
    direction: str
    entry: float
    stop: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp: Optional[float] = None
    current_price: Optional[float] = None
    setup: Optional[str] = "default"
    quality: Optional[str] = "A"
    regime_score: Optional[int] = 0
    atr_expanding: Optional[Any] = False
    risk_per_unit: Optional[float] = None
    atr_value: Optional[float] = None
    runner_config: Optional[RunnerConfig] = Field(default_factory=RunnerConfig)
    trend_state: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class PriceUpdatePayload(BaseModel):
    secret: Optional[str] = None
    event: str = "PRICE_UPDATE"
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

    class Config:
        extra = "allow"

# =============================================================
# TELEGRAM
# =============================================================
def send_telegram(text: str) -> bool:
    if not ENABLE_TELEGRAM:
        return False
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        response = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=8)
        return response.status_code == 200
    except Exception as exc:
        print(f"Telegram error: {exc}")
        return False


def telegram_trade_message(payload: TradeSignalPayload, decision: str, reason: str, risk_amount_gbp: Optional[float]) -> str:
    emoji = "✅" if decision == "ACTIVE" else "❌"
    lines = [
        f"{emoji} {decision} — {payload.instrument} {normalize_direction(payload.direction)}",
        f"Setup: {payload.setup} | Quality: {payload.quality}",
        f"Entry: {payload.entry} | SL: {payload.stop}",
        f"TP1: {payload.tp1} | TP2: {payload.tp2 or payload.tp}",
        f"Regime: R{payload.regime_score} | ATR expanding: {payload.atr_expanding}",
    ]
    if risk_amount_gbp is not None:
        lines.append(f"Risk: £{risk_amount_gbp}")
    lines.append(f"Reason: {reason}")
    lines.append(f"Time: {now()}")
    return "\n".join(lines)

# =============================================================
# FILTERS
# =============================================================
def evaluate_session(instrument: str) -> Tuple[str, str]:
    if not SESSION_FILTER_ENABLED:
        return "BYPASSED", "SESSION_FILTER_DISABLED"
    utc_hour = datetime.now(timezone.utc).hour
    if 12 <= utc_hour < 16:
        return "PRIME", "LONDON_NY_OVERLAP"
    if 7 <= utc_hour < 12:
        return "GOOD", "LONDON_SESSION"
    if 16 <= utc_hour < 21:
        return "GOOD", "NY_SESSION"
    return "POOR", "OFF_HOURS"


def session_allows_trade(session_quality: str) -> bool:
    if not SESSION_FILTER_ENABLED:
        return True
    return session_quality in {"PRIME", "GOOD", "BYPASSED"}


def compute_confidence(payload: TradeSignalPayload, session_quality: str) -> Tuple[int, str, List[str]]:
    score = 0
    reasons: List[str] = []

    quality = (payload.quality or "").upper()
    if quality == "A+":
        score += 40
        reasons.append("QUALITY_A_PLUS+40")
    elif quality == "A":
        score += 30
        reasons.append("QUALITY_A+30")

    regime = safe_int(payload.regime_score, 0)
    if regime >= 5:
        score += 25
        reasons.append("REGIME_STRONG+25")
    elif regime >= 3:
        score += 15
        reasons.append("REGIME_TRADEABLE+15")

    if str(payload.atr_expanding).lower() in {"true", "1"}:
        score += 10
        reasons.append("ATR_EXPANDING+10")

    if session_quality == "PRIME":
        score += 20
        reasons.append("SESSION_PRIME+20")
    elif session_quality == "GOOD":
        score += 10
        reasons.append("SESSION_GOOD+10")
    elif session_quality == "BYPASSED":
        reasons.append("SESSION_BYPASSED+0")

    mode = "HIGH" if score >= 70 else ("MEDIUM" if score >= 40 else "LOW")
    return min(score, 100), mode, reasons


def is_duplicate(trade_uid: str, instrument: str, direction: str, entry: float) -> bool:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=DUPLICATE_WINDOW_MINUTES)).isoformat()
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT id FROM signals
        WHERE event = 'TRADE_SIGNAL'
          AND (trade_uid = ? OR (instrument = ? AND direction = ? AND ABS(entry - ?) < 0.00001))
          AND created_at >= ?
        LIMIT 1
    """, (trade_uid, instrument, direction, entry, cutoff))
    row = cur.fetchone()
    con.close()
    return row is not None


def count_todays_active_trades() -> int:
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    con = db()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM trades WHERE status = 'ACTIVE' AND created_at >= ?", (start_of_day,))
    row = cur.fetchone()
    con.close()
    return row["c"] if row else 0

# =============================================================
# LOGGING / STORAGE
# =============================================================
def log_signal(payload: TradeSignalPayload, trade_uid: str, decision: str, reason: str, risk_amount_gbp: Optional[float], telegram_sent: bool):
    con = db()
    cur = con.cursor()
    raw = payload.model_dump_json()
    tp2_or_tp = payload.tp2 if payload.tp2 is not None else payload.tp
    cur.execute("""
        INSERT INTO signals
        (event, trade_uid, instrument, direction, setup, quality, decision, reason,
         regime_score, atr_expanding, entry, stop, tp, tp1, tp2, current_price,
         risk_per_unit, atr_value, risk_amount_gbp, runner_config, trend_state,
         raw_payload, telegram_sent, bar_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "TRADE_SIGNAL",
        trade_uid,
        payload.instrument,
        normalize_direction(payload.direction),
        payload.setup,
        payload.quality,
        decision,
        reason,
        safe_int(payload.regime_score, 0),
        1 if str(payload.atr_expanding).lower() in {"true", "1"} else 0,
        safe_float(payload.entry),
        safe_float(payload.stop),
        safe_float(tp2_or_tp),
        safe_float(payload.tp1),
        safe_float(tp2_or_tp),
        safe_float(payload.current_price if payload.current_price is not None else payload.entry),
        safe_float(payload.risk_per_unit, abs(payload.entry - payload.stop)),
        safe_float(payload.atr_value),
        risk_amount_gbp,
        payload.runner_config.model_dump_json() if payload.runner_config else "{}",
        safe_json_dumps(payload.trend_state),
        raw,
        1 if telegram_sent else 0,
        safe_int(payload.bar_time, 0),
        now(),
    ))
    con.commit()
    con.close()


def save_trade(payload: TradeSignalPayload, trade_uid: str, status: str, reason: str, risk_amount_gbp: float):
    con = db()
    cur = con.cursor()
    raw = payload.model_dump_json()
    tp2_or_tp = payload.tp2 if payload.tp2 is not None else payload.tp
    cur.execute("""
        INSERT OR IGNORE INTO trades
        (trade_uid, instrument, direction, entry, stop, tp, tp1, tp2, current_price,
         setup, quality, regime_score, atr_expanding, raw_payload, risk_amount_gbp,
         risk_per_unit, atr_value, runner_config, trend_state, status, reason, bar_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_uid,
        payload.instrument,
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
        1 if str(payload.atr_expanding).lower() in {"true", "1"} else 0,
        raw,
        risk_amount_gbp,
        safe_float(payload.risk_per_unit, abs(payload.entry - payload.stop)),
        safe_float(payload.atr_value),
        payload.runner_config.model_dump_json() if payload.runner_config else "{}",
        safe_json_dumps(payload.trend_state),
        status,
        reason,
        safe_int(payload.bar_time, 0),
        now(),
    ))
    con.commit()
    con.close()


def save_price_update(payload: PriceUpdatePayload) -> Dict[str, Any]:
    if WEBHOOK_SECRET and payload.secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="invalid secret")

    con = db()
    cur = con.cursor()
    raw = payload.model_dump_json()
    cur.execute("""
        INSERT INTO market_state
        (instrument, current_price, ema20, atr_value, rsi, atr_expanding, ema_slope,
         regime_score, current_15m_bias, latest_swing_low, latest_swing_high,
         raw_payload, bar_time, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            raw_payload=excluded.raw_payload,
            bar_time=excluded.bar_time,
            updated_at=excluded.updated_at
    """, (
        payload.instrument,
        safe_float(payload.current_price),
        safe_float(payload.ema20),
        safe_float(payload.atr_value),
        safe_float(payload.rsi),
        1 if str(payload.atr_expanding).lower() in {"true", "1"} else 0,
        safe_float(payload.ema_slope),
        safe_int(payload.regime_score, 0),
        payload.current_15m_bias,
        safe_float(payload.latest_swing_low),
        safe_float(payload.latest_swing_high),
        raw,
        safe_int(payload.bar_time, 0),
        now(),
    ))
    con.commit()
    con.close()
    return {"status": "PRICE_UPDATE_OK", "instrument": payload.instrument, "bar_time": payload.bar_time}

# =============================================================
# CORE HANDLERS
# =============================================================
def handle_trade_signal(payload: TradeSignalPayload) -> Dict[str, Any]:
    if WEBHOOK_SECRET and payload.secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="invalid secret")

    direction = normalize_direction(payload.direction)
    if payload.entry <= 0 or payload.stop <= 0:
        raise HTTPException(status_code=422, detail="entry/stop must be > 0")

    if payload.tp2 is None and payload.tp is None:
        raise HTTPException(status_code=422, detail="tp2 or tp is required")

    if payload.current_price is None:
        payload.current_price = payload.entry

    if payload.risk_per_unit is None:
        payload.risk_per_unit = abs(payload.entry - payload.stop)

    trade_uid = make_trade_uid(payload.instrument, direction, payload.entry, payload.setup or "default", payload.bar_time)
    risk_amount_gbp = round(ACCOUNT_BALANCE_GBP * (RISK_PER_TRADE_PCT / 100.0), 2)

    if is_duplicate(trade_uid, payload.instrument, direction, payload.entry):
        reason = "DUPLICATE_SIGNAL"
        sent = send_telegram(telegram_trade_message(payload, "DUPLICATE", reason, risk_amount_gbp))
        log_signal(payload, trade_uid, "DUPLICATE", reason, risk_amount_gbp, sent)
        return {"status": "DUPLICATE", "trade_uid": trade_uid, "reason": reason}

    session_quality, session_reason = evaluate_session(payload.instrument)
    score, mode, score_reasons = compute_confidence(payload, session_quality)

    if not session_allows_trade(session_quality):
        reason = f"SESSION_FILTER:{session_reason}"
        sent = send_telegram(telegram_trade_message(payload, "REJECTED", reason, risk_amount_gbp))
        log_signal(payload, trade_uid, "REJECTED", reason, risk_amount_gbp, sent)
        return {"status": "REJECTED", "trade_uid": trade_uid, "reason": reason}

    if count_todays_active_trades() >= DAILY_TRADE_LIMIT:
        reason = f"DAILY_LIMIT_REACHED:{DAILY_TRADE_LIMIT}"
        sent = send_telegram(telegram_trade_message(payload, "DAILY_LIMIT", reason, risk_amount_gbp))
        log_signal(payload, trade_uid, "DAILY_LIMIT", reason, risk_amount_gbp, sent)
        return {"status": "DAILY_LIMIT", "trade_uid": trade_uid, "reason": reason}

    if ENABLE_CONFIDENCE_GATE and REJECT_LOW_CONFIDENCE and score < CONFIDENCE_MIN:
        reason = f"LOW_CONFIDENCE:{score}<{CONFIDENCE_MIN}"
        sent = send_telegram(telegram_trade_message(payload, "LOW_CONFIDENCE", reason, risk_amount_gbp))
        log_signal(payload, trade_uid, "LOW_CONFIDENCE", reason, risk_amount_gbp, sent)
        save_trade(payload, trade_uid, "REJECTED", reason, 0.0)
        return {"status": "LOW_CONFIDENCE", "trade_uid": trade_uid, "reason": reason}

    reason = "ALL_FILTERS_PASSED"
    sent = send_telegram(telegram_trade_message(payload, "ACTIVE", reason, risk_amount_gbp))
    log_signal(payload, trade_uid, "ACTIVE", reason, risk_amount_gbp, sent)
    save_trade(payload, trade_uid, "ACTIVE", reason, risk_amount_gbp)

    return {
        "status": "ACTIVE",
        "trade_uid": trade_uid,
        "reason": reason,
        "instrument": payload.instrument,
        "direction": direction,
        "entry": payload.entry,
        "stop": payload.stop,
        "tp": payload.tp2 if payload.tp2 is not None else payload.tp,
        "tp1": payload.tp1,
        "tp2": payload.tp2 if payload.tp2 is not None else payload.tp,
        "current_price": payload.current_price,
        "risk_per_unit": payload.risk_per_unit,
        "atr_value": payload.atr_value,
        "risk_amount_gbp": risk_amount_gbp,
        "confidence_score": score,
        "confidence_mode": mode,
        "confidence_reasons": score_reasons,
    }


def normalize_raw_body(raw: Any) -> Dict[str, Any]:
    # Handles clean dict and double-encoded JSON strings from TradingView.
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
    event = str(payload.get("event", "")).upper().strip()

    if event == "TRADE_SIGNAL":
        return handle_trade_signal(TradeSignalPayload(**payload))

    if event == "PRICE_UPDATE":
        return save_price_update(PriceUpdatePayload(**payload))

    raise HTTPException(status_code=422, detail=f"unsupported event: {event}")

# =============================================================
# ROUTES
# =============================================================
@app.on_event("startup")
def on_startup():
    init_db()
    migrate_db()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": now(),
        "version": "event_router_v1",
        "supported_events": ["TRADE_SIGNAL", "PRICE_UPDATE"],
        "session_filter_enabled": SESSION_FILTER_ENABLED,
        "account_balance_gbp": ACCOUNT_BALANCE_GBP,
        "risk_per_trade_pct": RISK_PER_TRADE_PCT,
    }


@app.post("/webhook/tradingview")
async def webhook_tradingview(request: Request):
    return await route_webhook(request)


@app.post("/webhook/entry")
async def webhook_entry(request: Request):
    return await route_webhook(request)


@app.get("/trades")
def get_trades(status: Optional[str] = Query(default="ACTIVE"), limit: int = 200):
    con = db()
    cur = con.cursor()
    if status is None or status == "":
        cur.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
    else:
        cur.execute("SELECT * FROM trades WHERE status = ? ORDER BY id DESC LIMIT ?", (status, limit))
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


@app.get("/market-state")
def get_market_state(instrument: Optional[str] = None):
    con = db()
    cur = con.cursor()
    if instrument:
        cur.execute("SELECT * FROM market_state WHERE instrument = ?", (instrument,))
        row = cur.fetchone()
        con.close()
        return dict(row) if row else {}
    cur.execute("SELECT * FROM market_state ORDER BY updated_at DESC")
    rows = [dict(row) for row in cur.fetchall()]
    con.close()
    return rows


@app.get("/signals/rejected")
def get_rejected_signals(limit: int = 100):
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM signals
        WHERE decision IN ('REJECTED','DAILY_LIMIT','LOW_CONFIDENCE','DUPLICATE','VALIDATION_ERROR')
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    con.close()
    return rows


@app.get("/signals/active")
def get_active_signals(limit: int = 100):
    con = db()
    cur = con.cursor()
    cur.execute("SELECT * FROM signals WHERE decision = 'ACTIVE' ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    con.close()
    return rows

# =============================================================
# LOCAL RUN
# =============================================================
if __name__ == "__main__":
    import uvicorn
    init_db()
    migrate_db()
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
