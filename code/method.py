from sanic import Sanic
from sanic.response import text

app = Sanic()

#####  Push your function below ######
@app.route("/users", methods=["POST",])
def create_user(request):
    return text("You are trying to create a user with the following POST: %s" % request.body)
######################################



if __name__=="__main__":
	app.run(host="0.0.0.0", port=8000, workers=4)
