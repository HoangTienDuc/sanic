from sanic import Sanic
from sanic.response import json

app = Sanic(__name__)


@app.route("/test_request")
async def test_request_args(request):
    return json({
        "parsed": True,
        "url": request.url,
        "query_string": request.query_string,
        "args": request.args,
        "raw_args": request.raw_args,
        "query_args": request.query_args,
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
