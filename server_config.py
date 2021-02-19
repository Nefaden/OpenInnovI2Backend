import os
from .app import app
from dotenv import load_dotenv
load_dotenv()

app.debug = False
#host = os.getenv("SERVER_HOST")
port = os.getenv("SERVER_PORT")
#app.run(host=host, port=port)
app.run(port=port)
