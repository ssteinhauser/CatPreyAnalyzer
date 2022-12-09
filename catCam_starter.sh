#!/usr/bin/env bash

# Telegram Bot info, edit for your bot
CHAT_ID="XXXXXXXXX"
BOT_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# Firebase info, edit to match your setup
FIREBASE_BUCKET=""

echo "Executing CatPreyAnalyzer"
# Tensorflow Stuff
PYTHONVERSION=$(ls -la `which python3`|awk -F '->' '{print $2}'|sed -e 's/^[[:space:]]*//')
export PYTHONPATH=$PYTHONPATH:$HOME/tensorflow/models/research:$HOME/tensorflow/models/research/slim:/usr/local/lib/$PYTHONVERSION/site-packages

export CHAT_ID
export BOT_TOKEN
export FIREBASE_BUCKET
cd $HOME/CatPreyAnalyzer
rm -f last_casc_img_*.jpg
rm -f live_img_*.jpg
# start software repeatedly, in case it crashes
while :; do
  python3 -u cascade.py
  echo "cascade software ended - restarting"
  sleep 2
done
