import requests

url = 'http://0.0.0.0:8001/'
files = {"image": open("indo_dl.jpg", "rb")}
params = {"export":"False", "search":"True", "add":"False", "remove":"False", "person_name":"duc"}
r = requests.post(url, params=params, files=files)
print(r.status_code)
if r.status_code == 200:
	print(r.text)
