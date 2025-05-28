import os

root = "./rainy/"

for file in os.listdir(root):
    with open(f'rainTrain.txt', 'a') as f:
        f.write(f'rainy/{file}\n')