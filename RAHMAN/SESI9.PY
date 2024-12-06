import imageio.v2 as img
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar dan mengkonversinya ke grayscale (jika perlu)
image = img.imread(r"C:\RAHMAN\rahman.jpg")

# Jika gambar RGB, konversi ke grayscale
if len(image.shape) == 3:
    image = np.dot(image[... , :3], [0.2989, 0.5870, 0.1140])  # Menggunakan rumus konversi ke grayscale

# Sobel filter
sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobelY = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

# Padding gambar
imgPad = np.pad(image, pad_width=1, mode='constant', constant_values=0)

# Matriks untuk menyimpan hasil gradien
Gx = np.zeros_like(image)
Gy = np.zeros_like(image)

# Proses konvolusi
for y in range(1, imgPad.shape[0] - 1):
    for x in range(1, imgPad.shape[1] - 1):
        area = imgPad[y-1:y+2, x-1:x+2]
        Gx[y-1, x-1] = np.sum(area * sobelX)
        Gy[y-1, x-1] = np.sum(area * sobelY)

# Menghitung magnitudo gradien
G = np.sqrt(Gx**2 + Gy**2)

# Normalisasi hasil ke rentang 0-255
G = (G / G.max()) * 255
G = np.clip(G, 0, 255)
G = G.astype(np.uint8)

# Menampilkan hasil
plt.figure(figsize=(10, 10))

# Menampilkan gambar asli
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Menampilkan hasil gradien X
plt.subplot(2, 2, 2)
plt.imshow(Gx, cmap='gray')
plt.title("Sobel X Gradient")
plt.axis('off')

# Menampilkan hasil gradien Y
plt.subplot(2, 2, 3)
plt.imshow(Gy, cmap='gray')
plt.title("Sobel Y Gradient")
plt.axis('off')

# Menampilkan hasil magnitudo gradien
plt.subplot(2, 2, 4)
plt.imshow(G, cmap='gray')
plt.title("Edge Magnitude (Sobel Filter)")
plt.axis('off')

plt.tight_layout()
plt.show()
