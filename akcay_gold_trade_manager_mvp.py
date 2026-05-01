"""
AKÇAY Gold Trade Manager — Notification Only / Final v1.2

Core principles:
- TradingView sends entry and price_update alerts.
- Server creates and manages active trades.
- Server detects TP1 / TP2 / Stop / Runner Stop events.
- Telegram sends clear manual action notifications.
- No automatic order execution.
- Multiple active trades/runners can be tracked.
- Duplicate TP1/TP2/Runner messages are prevented per trade_id.

Run locally:
  pip install fastapi uvicorn httpx pydantic python-dotenv
  uvicorn akcay_gold_trade_manager_v12:app --host 0.0.0.0 --port 8000

Railway Procfile:
  web: uvicorn akcay_gold_trade_manager_mvp:app --host 0.0.0.0 --port $PORT

Required environment variables:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
  WEBHOOK_SECRET=...

Optional:
  ANTHROPIC_API_KEY=...
  CLAUDE_MODEL=claude-sonnet-4-20250514

Entry alert JSON examples:

LONG:
{"secret":"123abc456","event":"entry","instrument":"GOLD","direction":"BUY","entry":{{close}},"stop_distance":10,"current_price":{{close}},"setup":"LONG","quality":"A"}

SHORT:
{"secret":"123abc456","event":"entry","instrument":"GOLD","direction":"SELL","entry":{{close}},"stop_distance":10,"current_price":{{close}},"setup":"SHORT","quality":"A"}

BO LONG:
{"secret":"123abc456","event":"entry","instrument":"GOLD","direction":"BUY","entry":{{close}},"stop_distance":10,"current_price":{{close}},"setup":"BO_LONG","quality":"A+"}

BO SHORT:
{"secret":"123abc456","event":"entry","instrument":"GOLD","direction":"SELL","entry":{{close}},"stop_distance":10,"current_price":{{close}},"setup":"BO_SHORT","quality":"A+"}

Price update alert JSON:
{"secret":"123abc456","event":"price_update","instrument":"GOLD","current_price":{{close}}}
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Literal, Optional, Set

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="AKÇAY Gold Trade Manager — Notification Only")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")  # FIX: updated model ID
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

Direction = Literal["BUY", "SELL"]
TradeStatus = Literal["ACTIVE", "TP1_DONE", "RUNNER", "STOPPED", "CLOSED"]


class TradingViewAlert(BaseModel):
    secret: Optional[str] = None
    event: Literal["entry", "price_update", "manual_close"]
    instrument: str = Field(default="GOLD")
    direction: Optional[Direction] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    stop_distance: Optional[float] = None
    current_price: Optional[float] = None
    lot: Optional[float] = None
    setup: Optional[str] = None
    quality: Optional[str] = None
    alert_id: Optional[str] = None  # Optional idempotency key from TradingView if available.


@dataclass
class Trade:
    id: str
    instrument: str
    direction: Direction
    entry: float
    stop: float
    lot: float
    setup: str
    quality: str
    created_at: datetime
    fingerprint: str
    status: TradeStatus = "ACTIVE"
    tp1: float = 0.0
    tp2: float = 0.0
    runner_active: bool = False
    runner_stop: Optional[float] = None
    last_price: Optional[float] = None
    events_done: Set[str] = field(default_factory=set)


active_trades: Dict[str, Trade] = {}
processed_global_events: Set[str] = set()
# FIX: removed processed_entry_fingerprints — duplicate guard uses active_trades fingerprint check


def validate_secret(secret: Optional[str]) -> None:
    if WEBHOOK_SECRET and secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


def round_price(value: float) -> float:
    return round(float(value), 2)


def make_entry_fingerprint(alert: TradingViewAlert, stop_value: float) -> str:
    """
    Prevents duplicate entry alerts from creating duplicate active trades.
    If TradingView can send alert_id, that is preferred. Otherwise we build a practical fingerprint.
    """
    if alert.alert_id:
        return f"ALERT_ID:{alert.alert_id}"

    return "|".join(
        [
            alert.instrument.upper(),
            str(alert.direction),
            str(round_price(alert.entry or 0)),
            str(round_price(stop_value)),
            alert.setup or "UNKNOWN",
            alert.quality or "UNKNOWN",
        ]
    )


def calculate_levels(direction: Direction, entry: float, stop: float) -> tuple[float, float]:
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("Stop must be different from entry")

    if direction == "BUY":
        return round_price(entry + risk), round_price(entry + (2 * risk))

    return round_price(entry - risk), round_price(entry - (2 * risk))


def level_reached(direction: Direction, price: float, level: float, event_type: str) -> bool:
    if event_type in {"TP1", "TP2"}:
        return price >= level if direction == "BUY" else price <= level
    if event_type == "STOP":
        return price <= level if direction == "BUY" else price >= level
    raise ValueError(f"Unknown event_type: {event_type}")


def mark_event_once(trade: Trade, event_key: str) -> bool:
    """
    Returns True only the first time this event is processed for this trade.
    This prevents duplicate Telegram notifications.
    """
    global_key = f"{trade.id}:{event_key}"

    if event_key in trade.events_done:
        return False
    if global_key in processed_global_events:
        return False

    trade.events_done.add(event_key)
    processed_global_events.add(global_key)
    return True


async def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured. Message would be:\n", text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        response.raise_for_status()


async def claude_message(event_name: str, trade: Trade, price: Optional[float] = None) -> str:
    fallback = format_fallback_message(event_name, trade, price)
    if not ANTHROPIC_API_KEY:
        return fallback

    system_prompt = """
You are AKÇAY Gold Trade Manager — Notification Only.
You do not place trades. You do not suggest new entries.
You only format clear Telegram messages for an already active manual trade.
Be short, decisive, and practical.
Use this style:
- Event
- Action
- Stop/runner instruction
- Manual execution reminder
Never say you executed anything.
""".strip()

    user_prompt = f"""
Event: {event_name}
Trade ID: {trade.id}
Instrument: {trade.instrument}
Direction: {trade.direction}
Entry: {trade.entry}
Current price: {price}
Initial stop: {trade.stop}
Runner stop: {trade.runner_stop}
TP1: {trade.tp1}
TP2: {trade.tp2}
Lot: {trade.lot}
Setup: {trade.setup}
Quality: {trade.quality}
Status: {trade.status}

Create a concise Telegram notification.
""".strip()

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": 350,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"].strip()
    except Exception as exc:
        print(f"Claude failed, using fallback. Error: {exc}")
        return fallback


def format_fallback_message(event_name: str, trade: Trade, price: Optional[float]) -> str:
    if event_name == "ENTRY_CREATED":
        return (
            f"GOLD {trade.direction} trade active\n"
            f"Trade ID: {trade.id}\n"
            f"Setup: {trade.setup} | Quality: {trade.quality}\n"
            f"Entry: {trade.entry}\n"
            f"Stop: {trade.stop}\n"
            f"TP1: {trade.tp1} → close 30% manually\n"
            f"TP2: {trade.tp2} → close 30% manually\n"
            f"Runner: after TP2, keep final 40% with runner tracking\n"
            f"No auto execution. Manual only."
        )

    if event_name == "DUPLICATE_ENTRY_IGNORED":
        return (
            f"Duplicate GOLD {trade.direction} entry ignored\n"
            f"Existing Trade ID: {trade.id}\n"
            f"Entry: {trade.entry}\n"
            f"Setup: {trade.setup} | Quality: {trade.quality}\n"
            f"No new trade was created."
        )

    if event_name == "TP1_REACHED":
        return (
            f"TP1 reached for GOLD {trade.direction}\n"
            f"Trade ID: {trade.id}\n"
            f"Current price: {price}\n"
            f"Action: close 30% manually.\n"
            f"Move stop near breakeven: {trade.runner_stop}\n"
            f"Remaining position tracking active."
        )

    if event_name == "TP2_REACHED":
        return (
            f"TP2 reached for GOLD {trade.direction}\n"
            f"Trade ID: {trade.id}\n"
            f"Current price: {price}\n"
            f"Action: close another 30% manually.\n"
            f"Runner active: final 40% should be managed by runner stop.\n"
            f"Runner protection stop: {trade.runner_stop}"
        )

    if event_name == "RUNNER_STOP_REACHED":
        return (
            f"RUNNER STOP reached for GOLD {trade.direction}\n"
            f"Trade ID: {trade.id}\n"
            f"Current price: {price}\n"
            f"Action: close final 40% manually if not already closed.\n"
            f"Trade management complete. No re-entry unless a new valid alert appears."
        )

    if event_name == "STOP_REACHED":
        return (
            f"STOP reached for GOLD {trade.direction}\n"
            f"Trade ID: {trade.id}\n"
            f"Current price: {price}\n"
            f"Action: close manually if not already closed.\n"
            f"No revenge trade. Wait for next valid alert."
        )

    return f"Trade event: {event_name}\nTrade ID: {trade.id}\nTrade: {trade.instrument} {trade.direction}\nPrice: {price}\nManual execution only."


@app.get("/")
def root() -> dict:
    return {"status": "AKÇAY Gold Trade Manager running", "active_trades": len(active_trades)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "active_trades": len(active_trades)}


@app.get("/trades")
def list_trades() -> list[dict]:
    return [
        {
            "id": t.id,
            "instrument": t.instrument,
            "direction": t.direction,
            "entry": t.entry,
            "stop": t.stop,
            "tp1": t.tp1,
            "tp2": t.tp2,
            "lot": t.lot,
            "setup": t.setup,
            "quality": t.quality,
            "status": t.status,
            "runner_active": t.runner_active,
            "runner_stop": t.runner_stop,
            "last_price": t.last_price,
            "created_at": t.created_at.isoformat(),
            "events_done": sorted(t.events_done),
            "fingerprint": t.fingerprint,
        }
        for t in active_trades.values()
    ]


@app.post("/webhook/tradingview")
async def tradingview_webhook(alert: TradingViewAlert) -> dict:
    validate_secret(alert.secret)

    if alert.event == "entry":
        if alert.direction is None or alert.entry is None:
            raise HTTPException(status_code=400, detail="Entry alert requires direction and entry")

        computed_stop = alert.stop
        if computed_stop is None:
            if alert.stop_distance is None:
                raise HTTPException(status_code=400, detail="Entry alert requires either stop or stop_distance")
            computed_stop = alert.entry - alert.stop_distance if alert.direction == "BUY" else alert.entry + alert.stop_distance

        computed_stop = round_price(computed_stop)
        entry = round_price(alert.entry)
        fingerprint = make_entry_fingerprint(alert, computed_stop)

        # Duplicate entry guard: active_trades fingerprint check.
        for existing_trade in active_trades.values():
            if existing_trade.fingerprint == fingerprint and existing_trade.status not in {"STOPPED", "CLOSED"}:
                message = await claude_message("DUPLICATE_ENTRY_IGNORED", existing_trade, alert.current_price)
                await send_telegram(message)
                return {"ok": True, "duplicate_ignored": True, "trade_id": existing_trade.id}

        tp1, tp2 = calculate_levels(alert.direction, entry, computed_stop)
        trade = Trade(
            id=str(uuid.uuid4())[:8],
            instrument=alert.instrument.upper(),
            direction=alert.direction,
            entry=entry,
            stop=computed_stop,
            lot=alert.lot or 0.0,
            setup=alert.setup or "UNKNOWN",
            quality=alert.quality or "UNKNOWN",
            created_at=datetime.now(timezone.utc),
            tp1=tp1,
            tp2=tp2,
            last_price=alert.current_price,
            fingerprint=fingerprint,
        )
        active_trades[trade.id] = trade

        message = await claude_message("ENTRY_CREATED", trade, alert.current_price)
        await send_telegram(message)
        return {"ok": True, "trade_id": trade.id, "tp1": trade.tp1, "tp2": trade.tp2}

    if alert.event == "price_update":
        if alert.current_price is None:
            raise HTTPException(status_code=400, detail="Price update requires current_price")

        current_price = round_price(alert.current_price)
        triggered_events = []

        for trade in list(active_trades.values()):
            if trade.instrument != alert.instrument.upper():
                continue
            if trade.status in {"STOPPED", "CLOSED"}:
                continue

            trade.last_price = current_price

            # Stop logic: before runner, use initial stop. After runner, use runner_stop.
            active_stop = trade.runner_stop if trade.runner_active and trade.runner_stop is not None else trade.stop
            stop_event_name = "RUNNER_STOP_REACHED" if trade.runner_active else "STOP_REACHED"
            stop_event_key = "RUNNER_STOP" if trade.runner_active else "STOP"

            if level_reached(trade.direction, current_price, active_stop, "STOP"):
                if mark_event_once(trade, stop_event_key):
                    trade.status = "CLOSED" if trade.runner_active else "STOPPED"
                    message = await claude_message(stop_event_name, trade, current_price)
                    await send_telegram(message)
                    triggered_events.append({"trade_id": trade.id, "event": stop_event_key})
                continue

            if level_reached(trade.direction, current_price, trade.tp1, "TP1"):
                if mark_event_once(trade, "TP1"):
                    trade.status = "TP1_DONE"
                    # After TP1, protect remaining position near breakeven.
                    trade.runner_stop = round_price(trade.entry)
                    message = await claude_message("TP1_REACHED", trade, current_price)
                    await send_telegram(message)
                    triggered_events.append({"trade_id": trade.id, "event": "TP1", "new_stop": trade.runner_stop})

            if level_reached(trade.direction, current_price, trade.tp2, "TP2"):
                if mark_event_once(trade, "TP2"):
                    trade.status = "RUNNER"
                    trade.runner_active = True
                    # v1.2 simple runner protection: protect at TP1.
                    # v1.3 can replace this with structure/ATR trailing.
                    trade.runner_stop = round_price(trade.tp1)
                    message = await claude_message("TP2_REACHED", trade, current_price)
                    await send_telegram(message)
                    triggered_events.append({"trade_id": trade.id, "event": "TP2", "runner_stop": trade.runner_stop})

        return {"ok": True, "triggered_events": triggered_events}

    if alert.event == "manual_close":
        closed = []
        for trade in active_trades.values():
            if trade.instrument == alert.instrument.upper() and trade.status not in {"STOPPED", "CLOSED"}:
                trade.status = "CLOSED"
                trade.events_done.add("MANUAL_CLOSE")
                closed.append(trade.id)
        return {"ok": True, "closed": closed}

    raise HTTPException(status_code=400, detail="Unknown event")
