"""
AKÇAY Gold Trade Manager — Notification Only / MVP v1

Purpose:
- Receive TradingView alerts
- Create active trades
- Monitor incoming price updates
- Detect TP1 / TP2 / Stop / Runner events
- Ask Claude only to format concise decision messages
- Send Telegram notifications
- Never place trades automatically

Run:
  pip install fastapi uvicorn httpx pydantic python-dotenv
  uvicorn akcay_gold_trade_manager_mvp:app --host 0.0.0.0 --port 8000

Environment variables:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
  ANTHROPIC_API_KEY=...
  CLAUDE_MODEL=...   # optional
  WEBHOOK_SECRET=... # optional but recommended

TradingView entry alert JSON example:
{
  "secret": "your-secret",
  "event": "entry",
  "instrument": "GOLD",
  "direction": "SELL",
  "entry": 4610.16,
  "stop": 4620.00,
  "current_price": 4610.16,
  "lot": 0.10,
  "setup": "Breakdown",
  "quality": "A+"
}

TradingView price update JSON example:
{
  "secret": "your-secret",
  "event": "price_update",
  "instrument": "GOLD",
  "current_price": 4594.91
}
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="AKÇAY Gold Trade Manager — Notification Only")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

Direction = Literal["BUY", "SELL"]
TradeStatus = Literal["ACTIVE", "TP1_DONE", "TP2_DONE", "STOPPED", "CLOSED", "RUNNER"]


class TradingViewAlert(BaseModel):
    secret: Optional[str] = None
    event: Literal["entry", "price_update", "manual_close"]
    instrument: str = Field(default="GOLD")
    direction: Optional[Direction] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    stop_distance: Optional[float] = None  # optional: server computes stop from entry +/- distance
    current_price: Optional[float] = None
    lot: Optional[float] = None
    setup: Optional[str] = None
    quality: Optional[str] = None


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
    status: TradeStatus = "ACTIVE"
    tp1: float = 0.0
    tp2: float = 0.0
    runner_active: bool = False
    runner_stop: Optional[float] = None
    last_price: Optional[float] = None
    events_done: set[str] = field(default_factory=set)


active_trades: Dict[str, Trade] = {}


def validate_secret(secret: Optional[str]) -> None:
    if WEBHOOK_SECRET and secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


def calculate_levels(direction: Direction, entry: float, stop: float) -> tuple[float, float]:
    """
    TP logic:
    - Risk distance = abs(entry - stop)
    - TP1 = 1R
    - TP2 = 2R
    Runner begins after TP2
    """
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("Stop must be different from entry")

    if direction == "BUY":
        return entry + risk, entry + (2 * risk)
    return entry - risk, entry - (2 * risk)


def level_reached(direction: Direction, price: float, level: float, event_type: str) -> bool:
    if event_type in {"TP1", "TP2"}:
        return price >= level if direction == "BUY" else price <= level
    if event_type == "STOP":
        return price <= level if direction == "BUY" else price >= level
    raise ValueError(f"Unknown event_type: {event_type}")


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
Instrument: {trade.instrument}
Direction: {trade.direction}
Entry: {trade.entry}
Current price: {price}
Stop: {trade.stop}
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
            f"Entry: {trade.entry}\n"
            f"Stop: {trade.stop}\n"
            f"TP1: {trade.tp1} → close 30% manually\n"
            f"TP2: {trade.tp2} → close 30% manually\n"
            f"Runner: after TP2, keep final 40% with structure tracking\n"
            f"No auto execution. Manual only."
        )

    if event_name == "TP1_REACHED":
        return (
            f"TP1 reached for GOLD {trade.direction}\n"
            f"Current price: {price}\n"
            f"Action: close 30% manually.\n"
            f"Move stop near breakeven if structure allows.\n"
            f"Remaining position tracking active."
        )

    if event_name == "TP2_REACHED":
        return (
            f"TP2 reached for GOLD {trade.direction}\n"
            f"Current price: {price}\n"
            f"Action: close another 30% manually.\n"
            f"Runner active: final 40% should be managed by structure, not fixed £ target.\n"
            f"Runner protection stop: {trade.runner_stop}"
        )

    if event_name == "RUNNER_STOP_REACHED":
        return (
            f"RUNNER STOP reached for GOLD {trade.direction}\n"
            f"Current price: {price}\n"
            f"Action: close final 40% manually if not already closed.\n"
            f"Trade management complete. No re-entry unless a new valid alert appears."
        )

    if event_name == "STOP_REACHED":
        return (
            f"STOP reached for GOLD {trade.direction}\n"
            f"Current price: {price}\n"
            f"Action: close manually if not already closed.\n"
            f"No revenge trade. Wait for next valid alert."
        )

    return f"Trade event: {event_name}\nTrade: {trade.instrument} {trade.direction}\nPrice: {price}\nManual execution only."


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
            "last_price": t.last_price,
            "runner_stop": t.runner_stop,
            "created_at": t.created_at.isoformat(),
            "events_done": sorted(t.events_done),
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

        tp1, tp2 = calculate_levels(alert.direction, alert.entry, computed_stop)
        trade = Trade(
            id=str(uuid.uuid4())[:8],
            instrument=alert.instrument.upper(),
            direction=alert.direction,
            entry=alert.entry,
            stop=round(computed_stop, 2),
            lot=alert.lot or 0.0,
            setup=alert.setup or "UNKNOWN",
            quality=alert.quality or "UNKNOWN",
            created_at=datetime.now(timezone.utc),
            tp1=round(tp1, 2),
            tp2=round(tp2, 2),
            last_price=alert.current_price,
        )
        active_trades[trade.id] = trade

        message = await claude_message("ENTRY_CREATED", trade, alert.current_price)
        await send_telegram(message)
        return {"ok": True, "trade_id": trade.id, "tp1": trade.tp1, "tp2": trade.tp2}

    if alert.event == "price_update":
        if alert.current_price is None:
            raise HTTPException(status_code=400, detail="Price update requires current_price")

        triggered_events = []
        for trade in list(active_trades.values()):
            if trade.instrument != alert.instrument.upper():
                continue
            if trade.status in {"STOPPED", "CLOSED"}:
                continue

            trade.last_price = alert.current_price

            active_stop = trade.runner_stop if trade.runner_active and trade.runner_stop is not None else trade.stop
            stop_event_name = "RUNNER_STOP_REACHED" if trade.runner_active else "STOP_REACHED"
            stop_event_key = "RUNNER_STOP" if trade.runner_active else "STOP"

            if stop_event_key not in trade.events_done and level_reached(trade.direction, alert.current_price, active_stop, "STOP"):
                trade.events_done.add(stop_event_key)
                trade.status = "STOPPED" if not trade.runner_active else "CLOSED"
                message = await claude_message(stop_event_name, trade, alert.current_price)
                await send_telegram(message)
                triggered_events.append({"trade_id": trade.id, "event": stop_event_key})
                continue

            if "TP1" not in trade.events_done and level_reached(trade.direction, alert.current_price, trade.tp1, "TP1"):
                trade.events_done.add("TP1")
                trade.status = "TP1_DONE"
                message = await claude_message("TP1_REACHED", trade, alert.current_price)
                await send_telegram(message)
                triggered_events.append({"trade_id": trade.id, "event": "TP1"})

            if "TP2" not in trade.events_done and level_reached(trade.direction, alert.current_price, trade.tp2, "TP2"):
                trade.events_done.add("TP2")
                trade.status = "RUNNER"
                trade.runner_active = True
                trade.runner_stop = round(trade.tp1, 2)  # simple v1.1 runner protection; later replace with structure stop
                message = await claude_message("TP2_REACHED", trade, alert.current_price)
                await send_telegram(message)
                triggered_events.append({"trade_id": trade.id, "event": "TP2", "runner_stop": trade.runner_stop})

        return {"ok": True, "triggered_events": triggered_events}

    if alert.event == "manual_close":
        closed = []
        for trade in active_trades.values():
            if trade.instrument == alert.instrument.upper() and trade.status not in {"STOPPED", "CLOSED"}:
                trade.status = "CLOSED"
                closed.append(trade.id)
        return {"ok": True, "closed": closed}

    raise HTTPException(status_code=400, detail="Unknown event")
