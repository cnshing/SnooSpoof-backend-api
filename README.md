# SnooSpoof
Generate Reddit posts of any users through HuggingFace's Transformers library


## Requirements
- An registered reddit web app
- Python 3.10.4 
- Docker or Poetry

##Setting up credentials
SnooSpoof gathers the latest post data of any user in order to finetune any pre-trained model. Therefore, we must supply
valid web app credentials to read data.

Clone the project and modify the client_id and client_secret to your corresponding web app credentials:
```
[SnooSpoof]
client_id=your_bot_client_id
client_secret=keep_your_bot_client_secret_secure
user_agent=SnooSpoof read-only at https://github.com/cnshing/SnooSpoof-backend-api
```

## Getting started - Docker
The fastest way to run the API is to build and run the Docker image.

Run the following command to build and start the container:
```bash
SNOOSPOOF_API_HOST=0.0.0.0
SNOOSPOOF_API_PORT=8123
sudo docker compose run --build --service-ports --detach api
```
where SNOOSPOOF_API_HOST and SNOOSPOOF_API_PORT is the host and port of choice

## Manual installation
It is possible to run the API without Docker. First install the dependencies with Poetry:

```
poetry install
```

Then link the credentials and move to the source directory:
```bash
ln -s "$PWD/praw.ini" ./src/SnooSpoof && cd ./src/SnooSpoof
```

and execute the following command:

```
uvicorn api.middleman:app --reload
```
