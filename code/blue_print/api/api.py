from sanic import Blueprint

import content
import info

api = Blueprint.group(content, info, url_prefix='/api')
