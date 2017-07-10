import os
import glob
import shutil

path = '_data'

if not os.path.exists(path):
    os.makedirs(path)

for fn in glob.glob("*/*"):
    file_name = os.path.split(fn)[-1]
    new_path = os.path.join(path, file_name)

    shutil.copy(fn, new_path)
    print("Copied file from %s to %s" % (fn, new_path))

print()
print("Extracted all files. Transfer all these files to the `data` directory")