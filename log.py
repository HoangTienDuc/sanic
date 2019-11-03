from sanic import Sanic
from sanic.log import logger
from sanic.response import text

app = Sanic('test')

@app.route('/')
async def test(request):
	logger.info("here is your log")
	return text("hello world!")

if __name__ == "__main__":
	app.run(debug=True, access_log=True)
