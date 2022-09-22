"""
Creating a docker image for the model
"""
FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN  pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

EXPOSE 8080

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8080",  "predict:app"]

