import time
from sanic import Sanic
from sanic.response import json
import asyncio

app = Sanic()

@app.websocket('/')
async def test(request, ws):
    while True:
        try:
            await asyncio.sleep(1)
            await ws.send('hello')
        except:
            print('closed')
            break

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
