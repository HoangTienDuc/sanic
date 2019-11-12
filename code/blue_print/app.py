from sanic import Sanic
import api

app = Sanic(__name__)

app.blueprint(api)
