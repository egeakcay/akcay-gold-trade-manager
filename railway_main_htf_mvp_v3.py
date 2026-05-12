# railway_main_htf_mvp_v3.py
# Production-safe MVP execution version
# TradingView webhook -> Railway -> filter/log -> Telegram -> /trades -> MT5 executor

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Any

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# =============================================================
# ENV / CONFIG
# =============================================================
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
DB_PATH = os.getenv("DB_PATH", "trades.db")

ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "true").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SESSION_FILTER_ENABLED = os.getenv("SESSION_FILTER_ENABLED", "true").lower() == "true"
ENABLE_HTF_GATE = os.getenv("ENABLE_HTF_GATE", "true").lower() == "true"
ENABLE_CONFIDENCE_GATE = os.getenv("ENABLE_CONFIDENCE_GATE", "false").lower() == "true"
REJECT_LOW_CONFIDENCE = os.getenv("REJECT_LOW_CONFIDENCE", "false").lower() == "true"

CONFIDENCE_MIN = int(os.getenv("CONFIDENCE_MIN", "60"))
DAILY_TRADE_LIMIT = int(os.getenv("DAILY_TRADE_LIMIT", "10"))
ACCOUNT_BALANCE_GBP = float(os.getenv("ACCOUNT_BALANCE_GBP", "10000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))

DUPLICATE_WINDOW_MINUTES = int(os.getenv("DUPLICATE_WINDOW_MINUTES", "15"))

# HTF state codes
HTF_STATE_MAP = {
    0: "BEARISH_ONLY",
    1: "BULLISH_ONLY",
    2: "NEUTRAL_BLOCKED",
    3: "BOTH_ALLOWED",
}

# =============================================================
# APP
# =============================================================
app = FastAPI(title="HTF MVP v3", version="3.0.0")


# =============================================================
# UTILS
# =============================================================
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def make_trade_uid(instrument: str, direction: str, entry: float, setup: str) -> str:
    raw = f"{instrument}|{direction}|{round(entry, 5)}|{setup}|{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


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
            trade_uid TEXT,
            instrument TEXT,
            direction TEXT,
            setup TEXT,
            quality TEXT,
            decision TEXT,
            reason TEXT,
            session_quality TEXT,
            session_reason TEXT,
            regime_score INTEGER,
            atr_expanding INTEGER,
            htf_tf TEXT,
            htf_state_code INTEGER,
            htf_state TEXT,
            htf_score INTEGER,
            confidence_score INTEGER,
            confidence_mode TEXT,
            confidence_reasons TEXT,
            entry REAL,
            stop REAL,
            tp REAL,
            current_price REAL,
            raw_payload TEXT,
            telegram_sent INTEGER DEFAULT 0,
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
            session_reason TEXT,
            confidence_score INTEGER,
            confidence_mode TEXT,
            confidence_reasons TEXT,
            raw_payload TEXT,
            risk_amount_gbp REAL,
            status TEXT,
            reason TEXT,
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

    # signals
    add_column_if_missing("signals", "trade_uid", "TEXT")
    add_column_if_missing("signals", "instrument", "TEXT")
    add_column_if_missing("signals", "direction", "TEXT")
    add_column_if_missing("signals", "setup", "TEXT")
    add_column_if_missing("signals", "quality", "TEXT")
    add_column_if_missing("signals", "decision", "TEXT")
    add_column_if_missing("signals", "reason", "TEXT")
    add_column_if_missing("signals", "session_quality", "TEXT")
    add_column_if_missing("signals", "session_reason", "TEXT")
    add_column_if_missing("signals", "regime_score", "INTEGER")
    add_column_if_missing("signals", "atr_expanding", "INTEGER")
    add_column_if_missing("signals", "htf_tf", "TEXT")
    add_column_if_missing("signals", "htf_state_code", "INTEGER")
    add_column_if_missing("signals", "htf_state", "TEXT")
    add_column_if_missing("signals", "htf_score", "INTEGER")
    add_column_if_missing("signals", "confidence_score", "INTEGER")
    add_column_if_missing("signals", "confidence_mode", "TEXT")
    add_column_if_missing("signals", "confidence_reasons", "TEXT")
    add_column_if_missing("signals", "entry", "REAL")
    add_column_if_missing("signals", "stop", "REAL")
    add_column_if_missing("signals", "tp", "REAL")
    add_column_if_missing("signals", "current_price", "REAL")
    add_column_if_missing("signals", "raw_payload", "TEXT")
    add_column_if_missing("signals", "telegram_sent", "INTEGER DEFAULT 0")

    # trades
    add_column_if_missing("trades", "current_price", "REAL")
    add_column_if_missing("trades", "setup", "TEXT")
    add_column_if_missing("trades", "quality", "TEXT")
    add_column_if_missing("trades", "regime_score", "INTEGER")
    add_column_if_missing("trades", "atr_expanding", "INTEGER")
    add_column_if_missing("trades", "htf_tf", "TEXT")
    add_column_if_missing("trades", "htf_state_code", "INTEGER")
    add_column_if_missing("trades", "htf_state", "TEXT")
    add_column_if_missing("trades", "htf_score", "INTEGER")
    add_column_if_missing("trades", "session_quality", "TEXT")
    add_column_if_missing("trades", "session_reason", "TEXT")
    add_column_if_missing("trades", "confidence_score", "INTEGER")
    add_column_if_missing("trades", "confidence_mode", "TEXT")
    add_column_if_missing("trades", "confidence_reasons", "TEXT")
    add_column_if_missing("trades", "raw_payload", "TEXT")
    add_column_if_missing("trades", "risk_amount_gbp", "REAL")
    add_column_if_missing("trades", "status", "TEXT")
    add_column_if_missing("trades", "reason", "TEXT")

    con.commit()
    con.close()


# =============================================================
# PAYLOAD
# =============================================================
class EntryPayload(BaseModel):
    secret: Optional[str] = None
    event: Optional[str] = "entry"
    instrument: str
    direction: str
    entry: float
    stop: float
    tp: float
    current_price: Optional[float] = None
    setup: Optional[str] = "default"
    quality: Optional[str] = "A"
    regime_score: Optional[int] = 0
    atr_expanding: Optional[int] = 0
    htf_tf: Optional[str] = "120"
    htf_state_code: Optional[int] = 3
    htf_score: Optional[int] = 0

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
        response = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=8,
        )
        return response.status_code == 200
    except Exception as exc:
        print(f"Telegram error: {exc}")
        return False


def telegram_trade_message(
    payload: EntryPayload,
    decision: str,
    reason: str,
    htf_state: str,
    session_quality: str,
    confidence_score: Optional[int],
    confidence_mode: Optional[str],
) -> str:
    emoji = "✅" if decision == "ACTIVE" else "❌"
    lines = [
        f"{emoji} {decision} — {payload.instrument} {payload.direction}",
        f"Setup: {payload.setup} | Quality: {payload.quality}",
        f"Entry: {payload.entry} | SL: {payload.stop} | TP: {payload.tp}",
        f"HTF: {htf_state} | Session: {session_quality}",
        f"Confidence: {confidence_score} ({confidence_mode})",
        f"Reason: {reason}",
        f"Time: {now()}",
    ]
    return "\n".join(lines)


# =============================================================
# FILTERS
# =============================================================
def evaluate_session(instrument: str) -> (str, str):
    """Returns (session_quality, session_reason)."""
    if not SESSION_FILTER_ENABLED:
        return "BYPASSED", "SESSION_FILTER_DISABLED"

    utc_hour = datetime.now(timezone.utc).hour

    # London 7-16 UTC, NY 12-21 UTC, overlap 12-16 UTC = best
    if 12 <= utc_hour < 16:
        return "PRIME", "LONDON_NY_OVERLAP"
    if 7 <= utc_hour < 12:
        return "GOOD", "LONDON_SESSION"
    if 16 <= utc_hour < 21:
        return "GOOD", "NY_SESSION"
    if 0 <= utc_hour < 7:
        return "POOR", "ASIAN_SESSION"
    return "POOR", "OFF_HOURS"


def session_allows_trade(session_quality: str) -> bool:
    if not SESSION_FILTER_ENABLED:
        return True
    return session_quality in ("PRIME", "GOOD", "BYPASSED")


def evaluate_htf(direction: str, htf_state_code: int) -> (bool, str, str):
    """Returns (allowed, htf_state_name, reason)."""
    state_name = HTF_STATE_MAP.get(htf_state_code, "BOTH_ALLOWED")

    if not ENABLE_HTF_GATE:
        return True, state_name, "HTF_GATE_DISABLED"

    direction_upper = (direction or "").upper()

    if htf_state_code == 3:
        return True, state_name, "BOTH_ALLOWED"
    if htf_state_code == 2:
        return False, state_name, "HTF_NEUTRAL_BLOCKED"
    if htf_state_code == 1 and direction_upper in ("BUY", "LONG"):
        return True, state_name, "HTF_BULLISH_ALIGNED"
    if htf_state_code == 0 and direction_upper in ("SELL", "SHORT"):
        return True, state_name, "HTF_BEARISH_ALIGNED"
    return False, state_name, "HTF_CONFLICT"


def compute_confidence(payload: EntryPayload, session_quality: str, htf_aligned: bool) -> (int, str, List[str]):
    score = 0
    reasons: List[str] = []

    quality = (payload.quality or "").upper()
    if quality == "A":
        score += 30
        reasons.append("QUALITY_A+30")
    elif quality == "B":
        score += 15
        reasons.append("QUALITY_B+15")
    else:
        reasons.append("QUALITY_OTHER+0")

    regime = safe_int(payload.regime_score, 0)
    if regime >= 70:
        score += 20
        reasons.append("REGIME_STRONG+20")
    elif regime >= 40:
        score += 10
        reasons.append("REGIME_MID+10")
    else:
        reasons.append("REGIME_WEAK+0")

    if safe_int(payload.atr_expanding, 0) == 1:
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
    else:
        reasons.append("SESSION_POOR+0")

    if htf_aligned:
        score += 20
        reasons.append("HTF_ALIGNED+20")
    else:
        reasons.append("HTF_NOT_ALIGNED+0")

    mode = "HIGH" if score >= 70 else ("MEDIUM" if score >= 40 else "LOW")
    return min(score, 100), mode, reasons


def is_duplicate(trade_uid: str, instrument: str, direction: str, entry: float) -> bool:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=DUPLICATE_WINDOW_MINUTES)).isoformat()
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT id FROM signals
        WHERE (trade_uid = ?
               OR (instrument = ? AND direction = ? AND ABS(entry - ?) < 0.00001))
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
    cur.execute("""
        SELECT COUNT(*) AS c FROM trades
        WHERE status = 'ACTIVE' AND created_at >= ?
    """, (start_of_day,))
    row = cur.fetchone()
    con.close()
    return row["c"] if row else 0


# =============================================================
# LOGGING
# =============================================================
def log_signal(
    trade_uid: str,
    payload: EntryPayload,
    decision: str,
    reason: str,
    htf_state: str,
    session_quality: Optional[str],
    session_reason: Optional[str],
    confidence_score: Optional[int],
    confidence_mode: Optional[str],
    confidence_reasons: Optional[List[str]],
    telegram_sent: bool = False,
):
    raw_payload = payload.model_dump_json()
    con = db()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO signals
        (trade_uid, instrument, direction, setup, quality, decision, reason,
         session_quality, session_reason, regime_score, atr_expanding,
         htf_tf, htf_state_code, htf_state, htf_score,
         confidence_score, confidence_mode, confidence_reasons,
         entry, stop, tp, current_price, raw_payload, telegram_sent, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_uid,
        payload.instrument,
        payload.direction,
        payload.setup,
        payload.quality,
        decision,
        reason,
        session_quality,
        session_reason,
        safe_int(payload.regime_score, 0),
        safe_int(payload.atr_expanding, 0),
        str(payload.htf_tf or "120"),
        safe_int(payload.htf_state_code, 3),
        htf_state,
        safe_int(payload.htf_score, 0),
        confidence_score,
        confidence_mode,
        json.dumps(confidence_reasons or []),
        safe_float(payload.entry),
        safe_float(payload.stop),
        safe_float(payload.tp),
        safe_float(payload.current_price if payload.current_price is not None else payload.entry),
        raw_payload,
        1 if telegram_sent else 0,
        now(),
    ))
    con.commit()
    con.close()


def save_trade(
    trade_uid: str,
    payload: EntryPayload,
    status: str,
    reason: str,
    htf_state: str,
    session_quality: str,
    confidence_score: int,
    confidence_mode: str,
    risk_amount_gbp: float,
    session_reason: Optional[str] = None,
    confidence_reasons: Optional[List[str]] = None,
):
    raw_payload = payload.model_dump_json()
    con = db()
    cur = con.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO trades
        (trade_uid, instrument, direction, entry, stop, tp, current_price,
         setup, quality, regime_score, atr_expanding, htf_tf, htf_state_code,
         htf_state, htf_score, session_quality, session_reason,
         confidence_score, confidence_mode, confidence_reasons, raw_payload,
         risk_amount_gbp, status, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_uid,
        payload.instrument,
        payload.direction,
        safe_float(payload.entry),
        safe_float(payload.stop),
        safe_float(payload.tp),
        safe_float(payload.current_price if payload.current_price is not None else payload.entry),
        payload.setup,
        payload.quality,
        safe_int(payload.regime_score, 0),
        safe_int(payload.atr_expanding, 0),
        str(payload.htf_tf or "120"),
        safe_int(payload.htf_state_code, 3),
        htf_state,
        safe_int(payload.htf_score, 0),
        session_quality,
        session_reason,
        confidence_score,
        confidence_mode,
        json.dumps(confidence_reasons or []),
        raw_payload,
        risk_amount_gbp,
        status,
        reason,
        now(),
    ))
    con.commit()
    con.close()


# =============================================================
# CORE ENTRY HANDLER
# =============================================================
def entry_webhook(payload: EntryPayload) -> dict:
    # Secret check
    if WEBHOOK_SECRET and payload.secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="invalid secret")

    # Normalize defaults
    if payload.htf_state_code is None:
        payload.htf_state_code = 3
    if payload.htf_score is None:
        payload.htf_score = 0
    if payload.current_price is None:
        payload.current_price = payload.entry

    trade_uid = make_trade_uid(payload.instrument, payload.direction, payload.entry, payload.setup or "default")
    htf_state_name = HTF_STATE_MAP.get(safe_int(payload.htf_state_code, 3), "BOTH_ALLOWED")

    # ---------- 1. DUPLICATE CHECK ----------
    if is_duplicate(trade_uid, payload.instrument, payload.direction, payload.entry):
        reason = "DUPLICATE_SIGNAL"
        telegram_sent = send_telegram(
            telegram_trade_message(payload, "DUPLICATE", reason, htf_state_name, "UNKNOWN", None, None)
        )
        log_signal(
            trade_uid=trade_uid,
            payload=payload,
            decision="DUPLICATE",
            reason=reason,
            htf_state=htf_state_name,
            session_quality="UNKNOWN",
            session_reason="DUPLICATE_CHECK",
            confidence_score=None,
            confidence_mode=None,
            confidence_reasons=[],
            telegram_sent=telegram_sent,
        )
        return {"status": "DUPLICATE", "trade_uid": trade_uid, "reason": reason}

    # ---------- 2. SESSION ----------
    session_quality, session_reason = evaluate_session(payload.instrument)

    # ---------- 3. HTF ----------
    htf_allowed, htf_state_name, htf_reason = evaluate_htf(payload.direction, safe_int(payload.htf_state_code, 3))

    # ---------- 4. CONFIDENCE ----------
    score, mode, score_reasons = compute_confidence(payload, session_quality, htf_allowed)

    # ---------- 5. SESSION FILTER ----------
    if not session_allows_trade(session_quality):
        reason = f"SESSION_FILTER:{session_reason}"
        telegram_sent = send_telegram(
            telegram_trade_message(payload, "SESSION_FILTER", reason, htf_state_name, session_quality, score, mode)
        )
        log_signal(
            trade_uid=trade_uid,
            payload=payload,
            decision="SESSION_FILTER",
            reason=reason,
            htf_state=htf_state_name,
            session_quality=session_quality,
            session_reason=session_reason,
            confidence_score=score,
            confidence_mode=mode,
            confidence_reasons=score_reasons,
            telegram_sent=telegram_sent,
        )
        return {"status": "SESSION_FILTER", "trade_uid": trade_uid, "reason": reason}

    # ---------- 6. HTF GATE ----------
    if not htf_allowed:
        reason = f"HTF_CONFLICT:{htf_reason}"
        telegram_sent = send_telegram(
            telegram_trade_message(payload, "HTF_CONFLICT", reason, htf_state_name, session_quality, score, mode)
        )
        log_signal(
            trade_uid=trade_uid,
            payload=payload,
            decision="HTF_CONFLICT",
            reason=reason,
            htf_state=htf_state_name,
            session_quality=session_quality,
            session_reason=session_reason,
            confidence_score=score,
            confidence_mode=mode,
            confidence_reasons=score_reasons,
            telegram_sent=telegram_sent,
        )
        return {"status": "HTF_CONFLICT", "trade_uid": trade_uid, "reason": reason}

    # ---------- 7. DAILY LIMIT ----------
    todays = count_todays_active_trades()
    if todays >= DAILY_TRADE_LIMIT:
        reason = f"DAILY_LIMIT_REACHED:{todays}/{DAILY_TRADE_LIMIT}"
        telegram_sent = send_telegram(
            telegram_trade_message(payload, "DAILY_LIMIT", reason, htf_state_name, session_quality, score, mode)
        )
        log_signal(
            trade_uid=trade_uid,
            payload=payload,
            decision="DAILY_LIMIT",
            reason=reason,
            htf_state=htf_state_name,
            session_quality=session_quality,
            session_reason=session_reason,
            confidence_score=score,
            confidence_mode=mode,
            confidence_reasons=score_reasons,
            telegram_sent=telegram_sent,
        )
        return {"status": "DAILY_LIMIT", "trade_uid": trade_uid, "reason": reason}

    # ---------- 8. CONFIDENCE GATE ----------
    if ENABLE_CONFIDENCE_GATE and REJECT_LOW_CONFIDENCE and score < CONFIDENCE_MIN:
        reason = f"LOW_CONFIDENCE:{score}<{CONFIDENCE_MIN}"
        telegram_sent = send_telegram(
            telegram_trade_message(payload, "LOW_CONFIDENCE", reason, htf_state_name, session_quality, score, mode)
        )
        log_signal(
            trade_uid=trade_uid,
            payload=payload,
            decision="LOW_CONFIDENCE",
            reason=reason,
            htf_state=htf_state_name,
            session_quality=session_quality,
            session_reason=session_reason,
            confidence_score=score,
            confidence_mode=mode,
            confidence_reasons=score_reasons,
            telegram_sent=telegram_sent,
        )
        save_trade(
            trade_uid=trade_uid,
            payload=payload,
            status="REJECTED",
            reason=reason,
            htf_state=htf_state_name,
            session_quality=session_quality,
            confidence_score=score,
            confidence_mode=mode,
            risk_amount_gbp=0.0,
            session_reason=session_reason,
            confidence_reasons=score_reasons,
        )
        return {"status": "LOW_CONFIDENCE", "trade_uid": trade_uid, "reason": reason}

    # ---------- 9. ACTIVE ----------
    risk_amount = round(ACCOUNT_BALANCE_GBP * (RISK_PER_TRADE_PCT / 100.0), 2)
    reason = "ALL_FILTERS_PASSED"

    telegram_sent = send_telegram(
        telegram_trade_message(payload, "ACTIVE", reason, htf_state_name, session_quality, score, mode)
    )
    log_signal(
        trade_uid=trade_uid,
        payload=payload,
        decision="ACTIVE",
        reason=reason,
        htf_state=htf_state_name,
        session_quality=session_quality,
        session_reason=session_reason,
        confidence_score=score,
        confidence_mode=mode,
        confidence_reasons=score_reasons,
        telegram_sent=telegram_sent,
    )
    save_trade(
        trade_uid=trade_uid,
        payload=payload,
        status="ACTIVE",
        reason=reason,
        htf_state=htf_state_name,
        session_quality=session_quality,
        confidence_score=score,
        confidence_mode=mode,
        risk_amount_gbp=risk_amount,
        session_reason=session_reason,
        confidence_reasons=score_reasons,
    )

    return {
        "status": "ACTIVE",
        "trade_uid": trade_uid,
        "reason": reason,
        "instrument": payload.instrument,
        "direction": payload.direction,
        "entry": payload.entry,
        "stop": payload.stop,
        "tp": payload.tp,
        "current_price": payload.current_price,
        "htf_state": htf_state_name,
        "session_quality": session_quality,
        "confidence_score": score,
        "confidence_mode": mode,
        "risk_amount_gbp": risk_amount,
    }


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
        "htf_gate": ENABLE_HTF_GATE,
        "confidence_gate": ENABLE_CONFIDENCE_GATE,
        "reject_low_confidence": REJECT_LOW_CONFIDENCE,
        "session_filter_enabled": SESSION_FILTER_ENABLED,
        "account_balance_gbp": ACCOUNT_BALANCE_GBP,
        "risk_per_trade_pct": RISK_PER_TRADE_PCT,
        "default_risk_gbp": DEFAULT_RISK_GBP,
    }


@app.post("/webhook/entry")
def webhook_entry(payload: EntryPayload):
    return entry_webhook(payload)


@app.post("/webhook/tradingview")
def webhook_tradingview(payload: EntryPayload):
    return entry_webhook(payload)


@app.get("/trades")
def get_trades(status: Optional[str] = Query(default="ACTIVE"), limit: int = 200):
    con = db()
    cur = con.cursor()
    if status is None or status == "":
        cur.execute("""
            SELECT * FROM trades
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
    else:
        cur.execute("""
            SELECT * FROM trades
            WHERE status = ?
            ORDER BY id DESC
            LIMIT ?
        """, (status, limit))
    rows = [dict(row) for row in cur.fetchall()]
    con.close()
    return rows


@app.get("/signals")
def get_signals(limit: int = 100):
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM signals
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    con.close()
    return rows


@app.get("/signals/rejected")
def get_rejected_signals(limit: int = 100):
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM signals
        WHERE decision IN ('REJECTED','SESSION_FILTER','HTF_CONFLICT','DAILY_LIMIT','LOW_CONFIDENCE','DUPLICATE')
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
    cur.execute("""
        SELECT * FROM signals
        WHERE decision = 'ACTIVE'
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
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
