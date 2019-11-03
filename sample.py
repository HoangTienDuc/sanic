from sanic import Sanic
from sanic.response import text

app = Sanic()

#####  Push your function below ######


######################################



if __name__=="__main__":
	app.run(host="0.0.0.0", port=8000, workers=4)
