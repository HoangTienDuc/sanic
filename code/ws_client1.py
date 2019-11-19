import asyncio
import websockets

def go():
    with websockets.connect('ws://localhost:8001/feed') as ws:
        ws.send(10)
        ws.recv()

loop = asyncio.get_event_loop()
loop.run_until_complete(go())

