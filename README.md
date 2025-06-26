# [M0YCX] Jupyter Notebook studies in Amateur Radio

> Studies in RF Electronics for Amateur Radio...

## Prereqs
The interactive notebooks will not render in github and will need to be
run from a Jupyter Lab server (see Docker Compose below...).

To run the Jupyter Lab service you will need to install:-

* [Docker](https://docs.docker.com/engine/install/)
* [Docker Compose](https://docs.docker.com/compose/install/)

Alternatively see https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html,
the prereq python libs for the notebooks are listed in `requirements.txt`.

## Startup the notebook service with docker compose

```
$ docker compose up --build
```

Access with browser http://127.0.0.1:8889

(see compose.yaml for the default login token and options)