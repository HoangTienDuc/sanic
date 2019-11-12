from sanic import Blueprint
import my_blueprint
import bp_text

content = Blueprint.group(my_blueprint, bp_text, url_prefix='/content')
