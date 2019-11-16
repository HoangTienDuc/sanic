from aiofiles import os as async_os
from sanic.response import file_stream
from sanic import Sanic

app = Sanic()

@app.route("/")
async def index(request):
    file_path = "/home/tienduchoang/Pictures/1.png"

    file_stat = await async_os.stat(file_path)
    headers = {"Content-Length": str(file_stat.st_size)}

    return await file_stream(
        file_path,
        headers=headers,
        chunked=False,
    )
if __name__ =="__main__":
	app.run(host="0.0.0.0", port=8002, workers=4)
