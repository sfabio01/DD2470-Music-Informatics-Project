from tqdm import tqdm
import torch
import numpy as np

CHUNK_SIZE = 1024 * 2048 * 3


mm = np.memmap("fma_processed/memmap.dat", dtype="float16", mode="r")
tm = torch.from_file("fma_processed/memmap.dat", dtype=torch.float16, size=7997*1024*2048*3) 
for i in tqdm(range(0, 1000)):
    chunk = mm[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
    m = chunk.mean()
