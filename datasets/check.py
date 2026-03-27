import os

# Replace 'your_file.txt' with the path to your text file
file_path = '/opt/data/private/szj/AAAI/UniFuseOurs/datasets/structure3d_train.txt'

# Open the text file and read the lines
with open(file_path, 'r') as file:
    lines = file.readlines()

# Split each line into two paths, check existence of each, and print if not exists
for line in lines:
    path1, path2 = line.strip().split(' ')  # Split by space and remove whitespace
    if not os.path.exists('/opt/data/private/szj/data/Structured3D/'+path1):
        print(path1)
    if not os.path.exists('/opt/data/private/szj/data/Structured3D/'+path2):
        print(path2)
