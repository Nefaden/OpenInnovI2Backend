import os
from .app import app


app.debug = False
#host = os.environ.get('IP', 'O.O.O.O')
port = int(os.environ.get("PORT", 5000))
#app.run(host='0.0.0.0', port=port)
app.run(port=port)
