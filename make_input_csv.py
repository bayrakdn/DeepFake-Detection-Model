import os
print("Running make_input_csv.py, cwd =", os.getcwd())
import pandas as pd
dirs = ["generated_videos", "real_videos"]

all_files = []
for d in dirs:
    path = os.path.join(os.getcwd(), d)
    if os.path.isdir(path):
        all_files += sorted(os.listdir(path))

df = pd.DataFrame(all_files, columns=["file_name"])
df.to_csv("input.csv", index=False)
print(f"Wrote input.csv with {len(all_files)} entries")
