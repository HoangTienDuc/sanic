from sanic.response import json
from sanic import Blueprint

bp = Blueprint('my bp')

@bp.route('/text_bp')
async def bp_text(request):
	return json({"text": "halo"})
