from sanic import Sanic
from my_blueprint import group

app = Sanic(__name__)
app.blueprint(group)

app.run(host='0.0.0.0', port=8000, debug=True)
