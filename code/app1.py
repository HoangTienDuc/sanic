from sanic import Sanic
from sanic.response import text

app = Sanic()

app.route("/sms")
async def hello(request):
	message_body = request.form["Body"][0]
	return text("You said: {}".format(message_body))


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8000)
