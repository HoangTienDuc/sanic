wrk.method = "POST"
wrk.headers["Content-Type"] = "multipart/form-data; boundary=----WebKitFormBoundaryX3bY6PBMcxB1vCan"
bodyhead = "------WebKitFormBoundaryX3bY6PBMcxB1vCan"
bodyhead = bodyhead .. '\r\n'
bodyhead = bodyhead .. [[Content-Disposition: form-data; name="image"; filename="image1.jpg"]]
bodyhead = bodyhead .. '\r\n'
bodyhead = bodyhead .. 'Content-Type: image/jpeg'
bodyhead = bodyhead .. '\r\n'
bodyhead = bodyhead .. '\r\n'

filename = "misc/old/tuyen.jpg"
file = io.open(filename, "rb")
bodyhead = bodyhead .. file:read("*a")
bodyhead = bodyhead .. '\r\n'
bodyhead = bodyhead .. '------WebKitFormBoundaryX3bY6PBMcxB1vCan--'
wrk.body   = bodyhead
io.close(file)
