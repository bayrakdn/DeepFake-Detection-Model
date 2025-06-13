import os
import pandas as pd

print("Running in:", os.getcwd())

folders = [
    ("app/generated_videos", 1),  
    ("app/real_videos",      0),  
]


rows = []
for folder, label in folders:
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".mp4"):
            rows.append((fname, label))

df = pd.DataFrame(rows, columns=["file_name","Is_fake"])
df.to_csv("dataset.csv", sep=";", index=False)
print(f"Wrote dataset.csv with {len(df)} entries")
