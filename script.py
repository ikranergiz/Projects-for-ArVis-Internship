import os
import shutil

# Saving frames which are located in different files names like exp1, ex2, exp3 in a file.

for i in range(71,91):
    image_path = f"CRASH/exp{i}/image0.jpg"
    os.rename(image_path, f"CRASH/exp{i}/image{i}.jpg")
    print(image_path)
    image_path = f"CRASH/exp{i}/image{i}.jpg"
    shutil.move(image_path,f'CRASH\photos\image{i - 2}.jpg')
    shutil.rmtree(f"CRASH/exp{i}")


