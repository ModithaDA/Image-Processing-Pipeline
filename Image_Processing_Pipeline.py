import cv2
import matplotlib.pyplot as plt

# Task 1: Read the image
image_path = '/content/WhatsApp Image 2024-11-07 at 23.19.11_a3666f99.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Task 2: Convert the image to grayscale and apply transformation crop, rotate and resize
# Display grayscale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(6, 6))
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# Crop the grayscale image
x_start, y_start, x_end, y_end = 0, 0, 4000, 1000
cropped_image = gray_image[y_start:y_end, x_start:x_end]
plt.figure(figsize=(6, 6))
plt.imshow(cropped_image, cmap='gray')
plt.title("Cropped Image")
plt.axis('off')
plt.show()

# Rotate the grayscale image
(h, w) = gray_image.shape[:2]
center = (w // 2, h // 2)
rotate_matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_image = cv2.warpAffine(gray_image, rotate_matrix, (w, h))
plt.figure(figsize=(6, 6))
plt.imshow(rotated_image, cmap='gray')
plt.title("Rotated Image")
plt.axis('off')
plt.show()

# Resize the grayscale image
resized_image = cv2.resize(gray_image, (100, 100))
plt.figure(figsize=(6, 6))
plt.imshow(resized_image, cmap='gray')
plt.title("Resized Image")
plt.axis('off')
plt.show()

# Task 3: Apply basic filter like Gaussian Blur and edge detection
#Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
plt.figure(figsize=(6, 6))
plt.imshow(blurred_image, cmap='gray')
plt.title("Gaussian Blurred Image")
plt.axis('off')
plt.show()

#Canny Edge Detection
edges = cv2.Canny(gray_image, 400, 800)
plt.figure(figsize=(6, 6))
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")
plt.axis('off')
plt.show()