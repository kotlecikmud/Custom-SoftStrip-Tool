"""
---ABOUT---

Script Name: cauzin_softstrip.py
Author: Filip Pawłowski
Contact: filippawlowski2012@gmail.com
"""

__version__ = "00.01.00.00"

from PIL import Image, ImageDraw
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2

# Constants
SYNC_HEADER = '10101010' * 2
CHUNK_SIZE = 1024
MARGIN_SIZE = 40  # Pixels for margin around the data area
MARKER_SIZE = 20  # Size of corner markers
CALIBRATION_STRIP_HEIGHT = 10  # Height of calibration strip


def add_alignment_markers(img):
    """Add corner markers and calibration strip to the image"""
    width, height = img.size
    new_width = width + (MARGIN_SIZE * 2)
    new_height = height + (MARGIN_SIZE * 2) + CALIBRATION_STRIP_HEIGHT

    # Create new image with margins
    new_img = Image.new('L', (new_width, new_height), 255)
    draw = ImageDraw.Draw(new_img)

    # Paste original image in center
    new_img.paste(img, (MARGIN_SIZE, MARGIN_SIZE))

    # Draw corner markers (L-shaped)
    def draw_L_marker(x, y, flip_x=False, flip_y=False):
        x1, y1 = x, y
        x2, y2 = x + (MARKER_SIZE if not flip_x else -MARKER_SIZE), y
        x3, y3 = x, y + (MARKER_SIZE if not flip_y else -MARKER_SIZE)

        draw.line([(x1, y1), (x2, y2)], fill=0, width=3)
        draw.line([(x1, y1), (x3, y3)], fill=0, width=3)

    # Draw corner markers
    draw_L_marker(MARGIN_SIZE // 2, MARGIN_SIZE // 2)  # Top-left
    draw_L_marker(new_width - MARGIN_SIZE // 2, MARGIN_SIZE // 2, flip_x=True)  # Top-right
    draw_L_marker(MARGIN_SIZE // 2, new_height - MARGIN_SIZE // 2 - CALIBRATION_STRIP_HEIGHT,
                  flip_y=True)  # Bottom-left
    draw_L_marker(new_width - MARGIN_SIZE // 2, new_height - MARGIN_SIZE // 2 - CALIBRATION_STRIP_HEIGHT, flip_x=True,
                  flip_y=True)  # Bottom-right

    # Add calibration strip at bottom
    strip_y = new_height - CALIBRATION_STRIP_HEIGHT
    strip_width = (new_width - MARGIN_SIZE * 2) // 4
    for i in range(4):
        x = MARGIN_SIZE + (i * strip_width)
        color = int(255 * (3 - i) / 3)  # White to black gradient
        draw.rectangle([x, strip_y, x + strip_width, new_height], fill=color)

    return new_img


def detect_and_transform(image):
    """Detect markers and transform image to correct perspective"""
    # Convert to numpy array and ensure grayscale
    img_array = np.array(image)
    if len(img_array.shape) > 2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Threshold the image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find L-shaped markers
    corners = []
    for contour in contours:
        if len(contour) > 10:  # Filter small contours
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # L-shaped markers should have 6 points
            if len(approx) == 6:
                # Get the centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    corners.append((cx, cy))

    # If we found all 4 corners
    if len(corners) == 4:
        # Sort corners to get them in correct order (top-left, top-right, bottom-left, bottom-right)
        corners = np.array(corners)
        center = np.mean(corners, axis=0)

        def sort_corners(corners, center):
            return [
                corners[np.argmin(np.linalg.norm(corners - center + [-1, -1], axis=1))],  # Top-left
                corners[np.argmin(np.linalg.norm(corners - center + [1, -1], axis=1))],  # Top-right
                corners[np.argmin(np.linalg.norm(corners - center + [-1, 1], axis=1))],  # Bottom-left
                corners[np.argmin(np.linalg.norm(corners - center + [1, 1], axis=1))]  # Bottom-right
            ]

        corners = sort_corners(corners, center)

        # Define destination points (rectangle)
        width = height = 800  # Fixed size output
        dst_points = np.array([
            [MARGIN_SIZE, MARGIN_SIZE],
            [width - MARGIN_SIZE, MARGIN_SIZE],
            [MARGIN_SIZE, height - MARGIN_SIZE],
            [width - MARGIN_SIZE, height - MARGIN_SIZE]
        ], dtype=np.float32)

        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(np.float32(corners), dst_points)

        # Apply transform
        transformed = cv2.warpPerspective(gray, matrix, (width, height))

        return Image.fromarray(transformed)

    return image


def generate_softstrip(data):
    # Original encoding logic remains the same until image creation
    chunks = []
    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i:i + CHUNK_SIZE]
        data_length = len(chunk)
        length_bits = format(data_length, '016b')
        chunk_bits = SYNC_HEADER + length_bits + ''.join(format(byte, '08b') for byte in chunk) + SYNC_HEADER
        chunks.append(chunk_bits)

    data_bits = ''.join(chunks)

    dibits = []
    for i in range(0, len(data_bits), 2):
        dibit = data_bits[i:i + 2]
        if dibit == '00':
            dibits.append(255)
        elif dibit == '01':
            dibits.append(192)
        elif dibit == '10':
            dibits.append(64)
        elif dibit == '11':
            dibits.append(0)

    width = 200
    height = (len(dibits) + width - 1) // width

    img = Image.new('L', (width, height), 255)
    pixels = img.load()

    for i, dibit_color in enumerate(dibits):
        x = i % width
        y = i // width
        pixels[x, y] = dibit_color

    # Add alignment markers and calibration strip
    return add_alignment_markers(img)


def decode_softstrip(image_path, output_file_path):
    # Load and process image
    img = Image.open(image_path)
    img = img.convert("L")

    # Detect markers and transform image
    img = detect_and_transform(img)

    # Remove margins
    width, height = img.size
    img = img.crop((MARGIN_SIZE, MARGIN_SIZE,
                    width - MARGIN_SIZE,
                    height - MARGIN_SIZE - CALIBRATION_STRIP_HEIGHT))

    # Convert to numpy array for processing
    pixels = np.array(img)

    # Color calibration using the calibration strip
    def calibrate_colors(color):
        # Map colors to nearest expected value
        if color > 224:  # White
            return 255
        elif color > 128:  # Light gray
            return 192
        elif color > 32:  # Dark gray
            return 64
        else:  # Black
            return 0

    # Apply calibration to all pixels
    calibrated_pixels = np.vectorize(calibrate_colors)(pixels)

    # Convert back to binary stream
    color_to_dibit = {
        255: '00',
        192: '01',
        64: '10',
        0: '11'
    }

    binary_stream = ''
    for row in calibrated_pixels:
        for color in row:
            if color in color_to_dibit:
                binary_stream += color_to_dibit[color]

    # Decode chunks
    decoded_data = bytearray()
    while binary_stream:
        start_idx = binary_stream.find(SYNC_HEADER)
        if start_idx == -1:
            break
        binary_stream = binary_stream[start_idx + len(SYNC_HEADER):]

        length_bits = binary_stream[:16]
        data_length = int(length_bits, 2)
        data_bits = binary_stream[16:16 + (data_length * 8)]

        for i in range(0, len(data_bits), 8):
            byte = data_bits[i:i + 8]
            decoded_data.append(int(byte, 2))

        binary_stream = binary_stream[16 + (data_length * 8) + len(SYNC_HEADER):]

    # Write decoded data
    with open(output_file_path, 'wb') as f:
        f.write(decoded_data)

    messagebox.showinfo("Success", f"Decoded data saved to {output_file_path}")


def select_file_to_encode():
    file_path = filedialog.askopenfilename(title="Select a text file to encode")
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        data = text.encode('utf-8')
        img = generate_softstrip(data)

        # Generate the output path with "_enc" suffix
        dir_name = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(dir_name, f"{base_name}_enc.png")

        # Save the encoded image
        img.save(save_path)
        messagebox.showinfo("Success", f"Softstrip image saved to {save_path}")


def select_file_to_decode():
    image_path = filedialog.askopenfilename(title="Select a Softstrip image to decode",
                                            filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg")])
    if image_path:
        # Generate the output path with "_dec" suffix
        dir_name = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(dir_name, f"{base_name}_dec.txt")

        # Decode and save the data
        decode_softstrip(image_path, save_path)


# GUI setup remains the same
root = tk.Tk()
root.title("Softstrip Encoder/Decoder")

encode_button = tk.Button(root, text="Encode Text File to Softstrip", command=select_file_to_encode)
encode_button.pack(pady=10)

decode_button = tk.Button(root, text="Decode Softstrip to Text File", command=select_file_to_decode)
decode_button.pack(pady=10)

root.mainloop()
