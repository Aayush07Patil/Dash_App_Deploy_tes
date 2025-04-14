#!/bin/bash
cd /home/site/wwwroot
export PYTHONPATH=/home/site/wwwroot:$PYTHONPATH
gunicorn --timeout 600 --bind=0.0.0.0:$PORT app:server