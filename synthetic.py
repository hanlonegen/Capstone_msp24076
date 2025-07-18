import cv2
import numpy as np
import random
import os

# Function to generate and draw curved scratches
def add_curved_scratches(image, width, height):
  # Random number of scratches
    x1, y1 = random.randint(0, width), random.randint(0, height)
    # Generate control points for a curve
    ctrl_x = x1 + random.randint(-50, 50)
    ctrl_y = y1 + random.randint(-50, 50)
    x2 = x1 + random.randint(20, 100) * random.choice([-1, 1])
    y2 = y1 + random.randint(20, 100) * random.choice([-1, 1])
    thickness = 1
    # Color range from white (255, 255, 255) to light gray (200, 200, 200)
    gray_level = random.randint(100, 200)
    color = (gray_level, gray_level, gray_level)
    
    # Create a quadratic Bezier curve
    points = np.array([[x1, y1], [ctrl_x, ctrl_y], [x2, y2]], np.int32)
    cv2.polylines(image, [points], False, color, thickness)

    blurred_scratches = cv2.GaussianBlur(image, (5, 5), random.randint(1, 3))
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0
    image[mask] = blurred_scratches[mask]
    return image

# Function to generate and draw scratches
def add_scratches(image, width, height):
    x1, y1 = random.randint(0, width), random.randint(0, height)
    # synthetic scratches 1, 2
    # x2 = x1 + random.randint(10, 100) * random.choice([-1, 1])
    # y2 = y1 + random.randint(10, 100) * random.choice([-1, 1])
    # synthetic scratches 3
    # x2 = x1 + random.randint(50, 300) * random.choice([-1, 1])
    # y2 = y1 + random.randint(50, 300) * random.choice([-1, 1])
    # synthetic scratches 4
    x2 = x1 + random.randint(200, 500) * random.choice([-1, 1])
    y2 = y1 + random.randint(200, 500) * random.choice([-1, 1])    

    thickness = 1

    # Color range from white (255, 255, 255) to light gray (200, 200, 200)
    gray_level = random.randint(100, 200)
    color = (gray_level, gray_level, gray_level)
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    blurred_scratches = cv2.GaussianBlur(image, (5, 5), random.randint(1, 3))
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0
    image[mask] = blurred_scratches[mask]

    return image

import os


FOLDER_PATH = "your_folder_path_here"  # Replace with your folder path containing images

num = 1

for _ in range(2):
    for file in os.listdir(FOLDER_PATH):
        if file.endswith('.jpg'):
            PATH = os.path.join(FOLDER_PATH, file)
            
        # Load base image
        base_image = cv2.imread(PATH)
        height, width = base_image.shape[:2]

        # Generate synthetic images

        scratched_image = base_image.copy()

        i = random.randint(0,2)
        j = random.randint(0,2)

        for _ in range(i):
            scratched_image = add_scratches(scratched_image, width, height)

        for _ in range(j):
            scratched_image = add_curved_scratches(scratched_image, width, height)

        # Save the image
        num_scratches = i + j

        if num_scratches != 0:
            if not os.path.exists('synthetic_images'):
                os.makedirs('synthetic_images')
            # Save the scratched image with a unique name
            cv2.imwrite(os.path.join('synthetic_images', f'{num}_scratches_{num_scratches}.jpg'), scratched_image)
            num += 1

