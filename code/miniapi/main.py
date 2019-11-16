import os
from sanic import Sanic
from sanic.response import text, json
import json

app = Sanic()

def pin(a, b, c):
    if a == True:
        print("1")
    if b == True:
        print("2")
    if c == True:
        print("3")

@app.route('/', methods=['POST'])
async def play(request):
    if 'image' in request.files:
        image = request.files.get('image')
        argss = request.args
        print(type(argss))
        print("request.args: ", argss['key'][0], len(argss))
        # print("agrss: ", agrss)
    return text("done")
if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000, workers=4)