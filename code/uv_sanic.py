import asyncio
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())



from motor.motor_asyncio import AsyncIOMotorClient

mongo_connection = AsyncIOMotorClient("<connection string>")

contacts = mongo_connection.mydatabase.contacts


from sanic import Sanic
from sanic.response import json

app = Sanic(__name__)


@app.route("/")
async def list(request):
    data = contacts.find().to_list(20)
    for x in data:
        x['id'] = str(x['_id'])
        del x['_id']

    return json(data)


@app.route("/new")
async def new(request):
    contact = request.json
    insert = contacts.insert_one(contact)
    return json({"inserted_id": str(insert.inserted_id)})


#loop = asyncio.get_event_loop()

app.run(host="0.0.0.0", port=8000, workers=3, debug=True)
