import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import json
import os,sys
os.chdir(sys.path[0])

with open('args.json') as f:
    config = json.load(f)


