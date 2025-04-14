#!/bin/bash
cd /home/site/wwwroot
export PYTHONPATH=/home/site/wwwroot:$PYTHONPATH
chmod +x /home/site/wwwroot/startup.sh
gunicorn --timeout 600 --bind=0.0.0.0:$PORT application:application