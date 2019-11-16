from sanic import Sanic
from sanic import response
from sanic.websocket import WebSocketProtocol

app = Sanic()

@app.websocket('/feed')
async def feed(request, ws):
    while True:
        data = 'hello!'
        print('Sending: ' + data)
        await ws.send(data)
        data = await ws.recv()
        print('Received: ' + data)

@app.route('/html2')
async def handle_request(request):
  return response.html("""<html><head><script>
         var exampleSocket = new WebSocket("ws://" + location.host + '/feed');
         exampleSocket.onmessage = function (event) {
         console.log(event.data)};</script></head><body><h1>Hello socket!</h1><p>hello</p></body></html>""")

app.run(host="0.0.0.0", port=8000)
