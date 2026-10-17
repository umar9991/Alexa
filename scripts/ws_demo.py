#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from websockets.asyncio.client import connect


async def run_demo(text: str, url: str) -> None:
    async with connect(url) as websocket:
        hello = await websocket.recv()
        print(hello)
        await websocket.send(json.dumps({"type": "user_message", "text": text}))
        while True:
            raw = await websocket.recv()
            print(raw)
            event = json.loads(raw)
            if event.get("type") == "approval_required":
                answer = await asyncio.to_thread(input, "Approve? [y/n] ")
                approved = answer.strip().lower().startswith("y")
                await websocket.send(
                    json.dumps(
                        {
                            "type": "approval",
                            "call_id": event["call_id"],
                            "approved": approved,
                        }
                    )
                )
            if event.get("type") == "run_finished":
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas WebSocket demo client")
    parser.add_argument("text", help="user message to send")
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws",
        help="backend WebSocket URL",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(args.text, args.url))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
