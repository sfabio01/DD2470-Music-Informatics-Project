# read the memmap file in chunks and save each channel as an image (spectrogram / chromagram, tempogram)

import numpy as np
from matplotlib import pyplot as plt

CHUNK_SIZE = 1024 * 2048 * 3

mm = np.memmap("fma_processed/memmap.dat", dtype="float16", mode="r")

img = mm[:CHUNK_SIZE].reshape(1024, 2048, 3)

# plot the img and save it to a file
plt.imshow(img[:, :, 0])
plt.savefig("spectrogram.png")

plt.imshow(img[:, :, 1])
plt.savefig("chromagram.png")

plt.imshow(img[:, :, 2])
plt.savefig("tempogram.png")
