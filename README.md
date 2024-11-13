# [M0YCX] Jupyter Notebook studies in Amateur Radio

> DISCLAIMER:-
>   Please don't use this content in any production sense, as I am
> developing this as a learning activity for my hobby in Amateur Radio...

## Prereqs
The interactive notebooks will not render in github and will need to be
run from a Jupyter Lab server (see Docker Compose below...).

To run the Jupyter Lab service you will need to install:-

* [Docker](https://docs.docker.com/engine/install/)
* [Docker Compose](https://docs.docker.com/compose/install/)

## Startup the notebook service with docker compose

```
$ docker compose up --build
```

Access with browser http://127.0.0.1:8889

(see compose.yaml for the default login token and options)