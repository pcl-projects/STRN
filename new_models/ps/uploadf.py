#! /usr/bin/env python3

from zeyu_utils import os as zos

ips = ["34.226.119.123", "54.173.237.88", "3.92.77.220"]

for ip in ips:
    cmd = f"scp -r ubuntu@{ip}:/home/ubuntu/sgd/training_logs/* ./logs/"
    cmd = f"scp -r *.py ubuntu@{ip}:/home/ubuntu/sgd/"
    # cmd = f"scp -r ubuntu@{ip}:/home/ubuntu/sgd/training_logs/* temp"
    zos.run_cmd(cmd)

# ps aux | grep local_update.py | grep -v grep | awk '{print $2}' | xargs kill 9
