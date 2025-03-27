# Intro

Mimic the new Sesame Demo.

# Building the container

Use the following command:

`docker build -t streaming-asr .`

# Running the code

There will be a `/models` folder in the repo. Using the `-v` switch, link the models folder with a volume to prevent continual downloading of the models. Use the following command to run locally:

`docker run --gpus all -p 0.0.0.0:8000:8000 -v "${PWD}\models:/app/models" -it streaming-asr`