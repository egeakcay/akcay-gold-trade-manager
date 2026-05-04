"""
AKÇAY Gold & Silver Trade Manager — Notification Only / MVP v1.3

Core principles:
- TradingView sends entry and price_update alerts.
- Server creates and manages active trades for GOLD and SILVER.
- Server uses the real TradingView SL when provided via `stop`.
- Server calculates TP1 / TP2 using R multiples from entry-stop distance.
- Server calculates position size using capital and instrument-specific risk.
- Telegram sends clear manual action notifications.
- No automatic order execution.
- Multiple active trades/runners can be tracked.
- Duplicate TP1/TP2/Runner messages are prevented per trade_id.

Run locally:
  pip install fastapi uvicorn httpx pydantic python-dotenv
  uvicorn akcay_gold_silver_trade_manager_mvp:app --host 0.0.0.0 --port 8000

Railway Procfile:
  web: uvicorn akcay_gold_silver_trade_manager_mvp:app --host 0.0.0.0 --port $PORT

Required Railway environment variables:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
  WEBHOOK_SECRET=...

Optional Railway environment variables:
  ANTHROPIC_API_KEY=...
  CLAUDE_MODEL=claude-sonnet-4-20250514
  CAPITAL_GBP=5000

Risk model:
  GOLD normal risk: 2% of capital = £100 if capital is £5000
  GOLD BO risk:     1.4% of capital = £70 if capital is £5000
  SILVER normal risk: 1% of capital = £50 if capital is £5000
  SILVER BO risk:     0.7% of capital = £35 if capital is £5000

TradingView entry alert examples:

GOLD LONG:
{"secret":"123abc456","event":"entry","instrument":"GOLD","direction":"BUY","entry":{{close}},"stop":{{plot("long_sl")}},"current_price":{{close}},"setup":"LONG","quality":"A"}

GOLD SHORT:
{"secret":"123abc456","event":"entry","instrument":"GOLD","direction":"SELL","entry":{{close}},"stop":{{plot("short_sl")}},"current_price":{{close}},"setup":"SHORT","quality":"A"}

GOLD BO LONG:
{"secret":"123abc456","event":"entry","instrument":"GOLD","direction":"BUY","entry":{{close}},"stop":{{plot("long_sl")}},"current_price":{{close}},"setup":"BO_LONG","quality":"A+"}

GOLD BO SHORT:
{"secret":"123abc456","event":"entry","instrument":"GOLD","direction":"SELL","entry":{{close}},"stop":{{plot("short_sl")}},"current_price":{{close}},"setup":"BO_SHORT","quality":"A+"}

SILVER LONG:
{"secret":"123abc456","event":"entry","instrument":"SILVER","direction":"BUY","entry":{{close}},"stop":{{plot("long_sl")}},"current_price":{{close}},"setup":"LONG","quality":"A"}

SILVER SHORT:
{"secret":"123abc456","event":"entry","instrument":"SILVER","direction":"SELL","entry":{{close}},"stop":{{plot("short_sl")}},"current_price":{{close}},"setup":"SHORT","quality":"A"}

SILVER BO LONG:
{"secret":"123abc456","event":"entry","instrument":"SILVER","direction":"BUY","entry":{{close}},"stop":{{plot("long_sl")}},"current_price":{{close}},"setup":"BO_LONG","quality":"A+"}

SILVER BO SHORT:
{"secret":"123abc456","event":"entry","instrument":"SILVER","direction":"SELL","entry":{{close}},"stop":{{plot("short_sl")}},"current_price":{{close}},"setup":"BO_SHORT","quality":"A+"}

GOLD price update:
{"secret":"123abc456","event":"price_update","instrument":"GOLD","current_price":{{close}}}

SILVER price update:
{"secret":"123abc456","event":"price_update","instrument":"SILVER","current_price":{{close}}}
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


app = FastAPI(title="AKÇAY Gold & Silver Trade Manager — Notification Only")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

CAPITAL_GBP = float(os.getenv("CAPITAL_GBP", "5000"))

# FIX 1: Minimum stop distance for SILVER to prevent unrealistically large position sizes.
# e.g. risk=£50, price_risk=0.01 → lot=5000 — operationally dangerous.
SILVER_MIN_STOP_DISTANCE = 0.05

INSTRUMENT_CONFIG = {
    "GOLD": {
        "capital_gbp": CAPITAL_GBP,
        "risk_per_trade": 0.02,
        "bo_risk_multiplier": 0.70,
        "tp1_r": 1.0,
        "tp2_r": 2.0,
        "round_decimals": 2,
    },
    "SILVER": {
        "capital_gbp": CAPITAL_GBP,
        "risk_per_trade": 0.01,
        "bo_risk_multiplier": 0.70,
        "tp1_r": 1.0,
        "tp2_r": 1.5,
        "round_decimals": 3,
    },
}

Direction = Literal["BUY", "SELL"]
TradeStatus = Literal["ACTIVE", "TP1_DONE", "RUNNER", "STOPPED", "CLOSED"]


class TradingViewAlert(BaseModel):
    secret: Optional[str] = None
    event: Literal["entry", "price_update", "manual_close"]
    instrument: str = Field(default="GOLD")
    direction: Optional[Direction] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    stop_distance: Optional[float] = None  # fallback only; prefer real TradingView stop
    current_price: Optional[float] = None
    lot: Optional[float] = None
    setup: Optional[str] = None
    quality: Optional[str] = None
    alert_id: Optional[str] = None


@dataclass
class Trade:
    id: str
    instrument: str
    direction: Direction
    entry: float
    stop: float
    lot: float
    risk_amount_gbp: float
    risk_percent: float
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


def validate_secret(secret: Optional[str]) -> None:
    if WEBHOOK_SECRET and secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


def get_config(instrument: str) -> dict:
    symbol = instrument.upper()
    if symbol not in INSTRUMENT_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unsupported instrument: {instrument}")
    return INSTRUMENT_CONFIG[symbol]


def is_bo_setup(setup: Optional[str]) -> bool:
    return bool(setup and setup.upper().startswith("BO"))


def round_price(value: float, instrument: str = "GOLD") -> float:
    config = get_config(instrument)
    return round(float(value), int(config["round_decimals"]))


def round_money(value: float) -> float:
    return round(float(value), 2)


def round_lot(value: float) -> float:
    return round(float(value), 2)


def make_entry_fingerprint(alert: TradingViewAlert, stop_value: float) -> str:
    """
    FIX 2: Quality removed from fingerprint.
    Same trade conditions (instrument|direction|entry|stop|setup) are treated as the same trade
    regardless of quality label variation (A vs A+).
    """
    if alert.alert_id:
        return f"ALERT_ID:{alert.alert_id}"

    instrument = alert.instrument.upper()
    return "|".join(
        [
            instrument,
            str(alert.direction),
            str(round_price(alert.entry or 0, instrument)),
            str(round_price(stop_value, instrument)),
            alert.setup or "UNKNOWN",
        ]
    )


def calculate_levels(direction: Direction, entry: float, stop: float, instrument: str) -> tuple[float, float]:
    config = get_config(instrument)
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("Stop must be different from entry")

    tp1_r = float(config["tp1_r"])
    tp2_r = float(config["tp2_r"])

    if direction == "BUY":
        return round_price(entry + (risk * tp1_r), instrument), round_price(entry + (risk * tp2_r), instrument)

    return round_price(entry - (risk * tp1_r), instrument), round_price(entry - (risk * tp2_r), instrument)


def calculate_position_size(instrument: str, entry: float, stop: float, setup: Optional[str]) -> tuple[float, float, float]:
    config = get_config(instrument)
    price_risk = abs(entry - stop)
    if price_risk <= 0:
        raise ValueError("Stop must be different from entry")

    base_risk_percent = float(config["risk_per_trade"])
    effective_risk_percent = base_risk_percent * float(config["bo_risk_multiplier"]) if is_bo_setup(setup) else base_risk_percent
    risk_amount_gbp = float(config["capital_gbp"]) * effective_risk_percent
    lot = risk_amount_gbp / price_risk

    return round_lot(lot), round_money(risk_amount_gbp), round(effective_risk_percent * 100, 2)


def level_reached(direction: Direction, price: float, level: float, event_type: str) -> bool:
    if event_type in {"TP1", "TP2"}:
        return price >= level if direction == "BUY" else price <= level
    if event_type == "STOP":
        return price <= level if direction == "BUY" else price >= level
    raise ValueError(f"Unknown event_type: {event_type}")


def mark_event_once(trade: Trade, event_key: str) -> bool:
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
You are AKÇAY Gold & Silver Trade Manager — Notification Only.
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
Lot/Units: {trade.lot}
Risk amount GBP: {trade.risk_amount_gbp}
Risk percent: {trade.risk_percent}%
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
            f"{trade.instrument} {trade.direction} trade active\n"
            f"Trade ID: {trade.id}\n"
            f"Setup: {trade.setup} | Quality: {trade.quality}\n"
            f"Entry: {trade.entry}\n"
            f"Stop: {trade.stop}\n"
            f"Risk: £{trade.risk_amount_gbp} ({trade.risk_percent}%)\n"
            f"Lot/Units: {trade.lot}\n"
            f"TP1: {trade.tp1} → close 30% manually\n"
            f"TP2: {trade.tp2} → close 30% manually\n"
            f"Runner: after TP2, keep final 40% with runner tracking\n"
            f"No auto execution. Manual only."
        )

    if event_name == "DUPLICATE_ENTRY_IGNORED":
        return (
            f"Duplicate {trade.instrument} {trade.direction} entry ignored\n"
            f"Existing Trade ID: {trade.id}\n"
            f"Entry: {trade.entry}\n"
            f"Setup: {trade.setup} | Quality: {trade.quality}\n"
            f"No new trade was created."
        )

    if event_name == "TP1_REACHED":
        return (
            f"TP1 reached for {trade.instrument} {trade.direction}\n"
            f"Trade ID: {trade.id}\n"
            f"Current price: {price}\n"
            f"Action: close 30% manually.\n"
            f"Close size: {round_lot(trade.lot * 0.30)}\n"
            f"Move stop near breakeven: {trade.runner_stop}\n"
            f"Remaining position tracking active."
        )

    if event_name == "TP2_REACHED":
        return (
            f"TP2 reached for {trade.instrument} {trade.direction}\n"
            f"Trade ID: {trade.id}\n"
            f"Current price: {price}\n"
            f"Action: close another 30% manually.\n"
            f"Close size: {round_lot(trade.lot * 0.30)}\n"
            f"Runner active: final 40% should be managed by runner stop.\n"
            f"Runner size: {round_lot(trade.lot * 0.40)}\n"
            f"Runner protection stop: {trade.runner_stop}"
        )

    if event_name == "RUNNER_STOP_REACHED":
        return (
            f"RUNNER STOP reached for {trade.instrument} {trade.direction}\n"
            f"Trade ID: {trade.id}\n"
            f"Current price: {price}\n"
            f"Action: close final 40% manually if not already closed.\n"
            f"Close size: {round_lot(trade.lot * 0.40)}\n"
            f"Trade management complete. No re-entry unless a new valid alert appears."
        )

    if event_name == "STOP_REACHED":
        return (
            f"STOP reached for {trade.instrument} {trade.direction}\n"
            f"Trade ID: {trade.id}\n"
            f"Current price: {price}\n"
            f"Action: close manually if not already closed.\n"
            f"Potential loss at planned risk: £{trade.risk_amount_gbp}\n"
            f"No revenge trade. Wait for next valid alert."
        )

    return f"Trade event: {event_name}\nTrade ID: {trade.id}\nTrade: {trade.instrument} {trade.direction}\nPrice: {price}\nManual execution only."


@app.get("/")
def root() -> dict:
    return {"status": "AKÇAY Gold & Silver Trade Manager running", "active_trades": len(active_trades)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "active_trades": len(active_trades), "capital_gbp": CAPITAL_GBP}


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
            "risk_amount_gbp": t.risk_amount_gbp,
            "risk_percent": t.risk_percent,
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

    instrument = alert.instrument.upper()
    get_config(instrument)

    if alert.event == "entry":
        if alert.direction is None or alert.entry is None:
            raise HTTPException(status_code=400, detail="Entry alert requires direction and entry")

        computed_stop = alert.stop
        if computed_stop is None:
            if alert.stop_distance is None:
                raise HTTPException(status_code=400, detail="Entry alert requires either stop or stop_distance")
            computed_stop = alert.entry - alert.stop_distance if alert.direction == "BUY" else alert.entry + alert.stop_distance

        entry = round_price(alert.entry, instrument)
        computed_stop = round_price(computed_stop, instrument)

        # FIX 1: SILVER minimum stop distance guard.
        # Prevents unrealistically large position sizes from tight stops.
        if instrument == "SILVER" and abs(entry - computed_stop) < SILVER_MIN_STOP_DISTANCE:
            msg = (
                f"SILVER trade rejected\n"
                f"Reason: stop distance too small (< {SILVER_MIN_STOP_DISTANCE})\n"
                f"Entry: {entry} | Stop: {computed_stop} | Distance: {round(abs(entry - computed_stop), 4)}"
            )
            await send_telegram(msg)
            return {"ok": False, "rejected": True, "reason": "SILVER_STOP_TOO_SMALL", "min_stop_distance": SILVER_MIN_STOP_DISTANCE}

        fingerprint = make_entry_fingerprint(alert, computed_stop)

        for existing_trade in active_trades.values():
            if existing_trade.fingerprint == fingerprint and existing_trade.status not in {"STOPPED", "CLOSED"}:
                message = await claude_message("DUPLICATE_ENTRY_IGNORED", existing_trade, alert.current_price)
                await send_telegram(message)
                return {"ok": True, "duplicate_ignored": True, "trade_id": existing_trade.id}

        tp1, tp2 = calculate_levels(alert.direction, entry, computed_stop, instrument)
        lot, risk_amount_gbp, risk_percent = calculate_position_size(instrument, entry, computed_stop, alert.setup)

        trade = Trade(
            id=str(uuid.uuid4())[:8],
            instrument=instrument,
            direction=alert.direction,
            entry=entry,
            stop=computed_stop,
            lot=lot,
            risk_amount_gbp=risk_amount_gbp,
            risk_percent=risk_percent,
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
        return {
            "ok": True,
            "trade_id": trade.id,
            "instrument": trade.instrument,
            "tp1": trade.tp1,
            "tp2": trade.tp2,
            "lot": trade.lot,
            "risk_amount_gbp": trade.risk_amount_gbp,
            "risk_percent": trade.risk_percent,
        }

    if alert.event == "price_update":
        if alert.current_price is None:
            raise HTTPException(status_code=400, detail="Price update requires current_price")

        current_price = round_price(alert.current_price, instrument)
        triggered_events = []

        for trade in list(active_trades.values()):
            if trade.instrument != instrument:
                continue
            if trade.status in {"STOPPED", "CLOSED"}:
                continue

            trade.last_price = current_price

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
                    trade.runner_stop = round_price(trade.entry, trade.instrument)
                    message = await claude_message("TP1_REACHED", trade, current_price)
                    await send_telegram(message)
                    triggered_events.append({"trade_id": trade.id, "event": "TP1", "new_stop": trade.runner_stop})

            if level_reached(trade.direction, current_price, trade.tp2, "TP2"):
                if mark_event_once(trade, "TP2"):
                    trade.status = "RUNNER"
                    trade.runner_active = True
                    trade.runner_stop = round_price(trade.tp1, trade.instrument)
                    message = await claude_message("TP2_REACHED", trade, current_price)
                    await send_telegram(message)
                    triggered_events.append({"trade_id": trade.id, "event": "TP2", "runner_stop": trade.runner_stop})

        return {"ok": True, "triggered_events": triggered_events}

    if alert.event == "manual_close":
        closed = []
        for trade in active_trades.values():
            if trade.instrument == instrument and trade.status not in {"STOPPED", "CLOSED"}:
                trade.status = "CLOSED"
                trade.events_done.add("MANUAL_CLOSE")
                closed.append(trade.id)
        return {"ok": True, "closed": closed}

    raise HTTPException(status_code=400, detail="Unknown event")