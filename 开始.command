#!/bin/bash

DIRNAME=$0
if [ "${DIRNAME:0:1}" = "/" ]; then
    CURDIR=$(dirname "$DIRNAME")
else
    CURDIR=$(pwd)/$(dirname "$DIRNAME")
fi

echo $CURDIR

$CURDIR/py310/bin/python $CURDIR/app.py --server_port 8081
