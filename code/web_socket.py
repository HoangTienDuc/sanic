from sanic import Sanic
from sanic.response import json
from sanic.websocket import WebSocketProtocol

app = Sanic()
@app.websocket('/feed')
async def feed(request, ws):
	while True:
		data = await ws.recv()
		print("Received: " + data)
		# if data = url "C://duc.mp4"
		# get file
		# handler - todo
#		x = 0
		for i in range(1000000000):
			return_data = str(data)
			await ws.send(return_data)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8000, protocol=WebSocketProtocol)
