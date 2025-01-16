FROM amd64/python:3.10.4-slim

COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ['uvicorn','simple_chatmodel:app','--host','0.0.0.0','--port','8000','-reload']
