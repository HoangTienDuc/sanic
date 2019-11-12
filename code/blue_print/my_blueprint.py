from sanic.response import json
from sanic import Blueprint

bp1 = Blueprint('bp1', url_prefix='/bp1')
bp2 = Blueprint('bp2', url_prefix='/bp2')
@bp1.route('/')
async def bp_root(request):
	return json({'my': 'blueprint'})
@bp2.route('/')
async def bp_text(request):
	return json({"haha": "hihi"})
group = Blueprint.group(bp1, bp2, url_prefix='/')
