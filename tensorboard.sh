#!/usr/bin/env bash

TF_LOGDIR=${TF_LOGDIR:=model}

exec pipenv run tensorboard --logdir=${TF_LOGDIR} --port=6006
