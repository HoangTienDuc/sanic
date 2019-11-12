from sanic import Blueprint

import static
import authors

content = Blueprint.group(static, authors, url_prefix='/content')
