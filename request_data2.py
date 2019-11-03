from sanic import Sanic
from sanic.response import json

app = Sanic()
@app.route('/files')
def post_json(request):
	test_file = request.files.get('test')
	file_parameters = {
		'body': test_file.body,
		'name': test_file.name,
		'type': test_file.type,
}
	return json({"received": True, "file_names": request.file.keys, "test_file_parameters": file_parameters})
if __name__=="__main__":
	app.run(host="0.0.0.0", port=8000)
