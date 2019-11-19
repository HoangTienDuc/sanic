import asyncio
import websockets

async def go():
    async with websockets.connect('ws://localhost:8000') as ws:
        for i in range(100):
            result = await ws.recv()
            print(result)

loop = asyncio.get_event_loop()
loop.run_until_complete(go())
