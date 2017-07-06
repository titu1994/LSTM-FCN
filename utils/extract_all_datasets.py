import os
import glob
import shutil

path = 'ALL_DATA'

if not os.path.exists(path):
    os.makedirs(path)

for fn in glob.glob("*/*"):
    file_name = os.path.split(fn)[-1]
    new_path = os.path.join(path, file_name)

    shutil.copy(fn, new_path)
    print("Copied file from %s to %s" % (fn, new_path))