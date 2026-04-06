"""
A script that downloads the American Standard Version bible from openbible.com,
cleans it and saves the result as a .txt file into the datasets/ folder.
"""

import urllib.request
import os

url = "https://openbible.com/textfiles/asv.txt"

with urllib.request.urlopen(url) as f:
    lines = str(f.read())
    lines = lines.split("\\n")
    lines = lines[2:-1] # delete the header and the closing "'"
    lines = [line.split("\\t")[1] for line in lines] # remove the verse indication that precedes every line

os.makedirs("datasets/", exist_ok=True)
with open("datasets/asv.txt", "w") as f:
    f.writelines(line + "\n" for line in lines)

print("Saved the American Standard Version bible into datasets/asv.txt")