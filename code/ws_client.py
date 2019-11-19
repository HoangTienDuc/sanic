
from contextlib import asynccontextmanager


url = "http://0.0.0.0:8000/feed"
@asynccontextmanager
async def websocket_connection(self, url):
    print("aaaaaaaaaaaaaaaaa")
    conn = await client.websocket(self, url)
    try:
        yield conn
    finally:
        await conn.close()

async def test_websocket_endpoint(data):
    async with client.websocket_connection() as websocket:
        message = data
        await websocket.send(message)
        resp = await websocket.recv()
        result = json.loads(resp)
#        assert result == {"event": "login", "error_code": 1000}
if __name__=="__main__":
	data = 10
	print(data)
	test_websocket_endpoint(data)
