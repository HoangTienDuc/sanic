from sanic import Sanic
from sanic.response import json
 
app = Sanic()

@app.route("/json")
def post_json(request):
	return json("Hello", "world!")

@app.route("/query_string")
def query_string(request):
	return json({"parsed": True, "args": request.args, "url": request.url, "query_string": request.query_string})

if __name__ == '__main__':
	app.run(host="0.0.0.0", port=8000, workers=5)
