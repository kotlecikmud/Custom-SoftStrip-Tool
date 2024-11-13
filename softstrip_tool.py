"""
---ABOUT---

Script Name: softstrip_tool.py
Author: Filip Pawłowski
Contact: filippawlowski2012@gmail.com
"""

__version__ = "00.02.00.00"

from PIL import Image
import os

while True:
    choice = input(f"Softstrip Tool - v{__version__}\nchoose: encode/decode >>>")

    if choice == "decode":
        def decode_softstrip(softstrip_image_path, img_width=256, padding=10):
            """
            Decodes binary data from a SoftStrip image format.

            :param softstrip_image_path: Path to the SoftStrip image to decode.
            :param img_width: Width of the SoftStrip image in pixels (should match encoding).
            :param padding: Number of pixels of padding around the SoftStrip code (should match encoding).
            :return: Decoded binary data.
            """
            # Open the SoftStrip image
            img = Image.open(softstrip_image_path).convert("1")  # Convert to binary (1-bit) mode
            img_width, img_height = img.size

            # Calculate the usable width and height for data, excluding padding
            usable_width = img_width - 2 * padding
            usable_height = img_height - 2 * padding

            # Decode binary data from image pixels
            data = bytearray()
            byte = 0
            bit_count = 0

            for y in range(padding, padding + usable_height):
                for x in range(padding, padding + usable_width):
                    # Check if the pixel is black (0) or white (1)
                    pixel = img.getpixel((x, y))
                    if pixel == 0:  # Black pixel represents a '1' bit
                        byte = (byte << 1) | 1
                    else:  # White pixel represents a '0' bit
                        byte = (byte << 1)

                    bit_count += 1
                    # Once we've collected 8 bits, add the byte to data
                    if bit_count == 8:
                        data.append(byte)
                        byte = 0
                        bit_count = 0

            print("Decoding complete.")
            return bytes(data)


        # Example usage
        softstrip_image_path = input("path/to/your/softstrip_image.png >>>")
        decoded_data = decode_softstrip(softstrip_image_path)

        # Save the decoded data to a file if needed
        output_file_path = "decoded_file.txt"
        with open(output_file_path, "wb") as file:
            file.write(decoded_data)
        print(f"Decoded data saved to {output_file_path}")

    elif choice == "encode":
        def encode_softstrip(input_file_path, output_image_path, img_width=256, padding=10):
            """
            Encodes binary data from a file into a SoftStrip image format.

            :param input_file_path: Path to the binary file to encode.
            :param output_image_path: Path to save the generated SoftStrip image.
            :param img_width: Width of the SoftStrip image in pixels.
            :param padding: Number of pixels of padding around the SoftStrip code.
            """
            # Read the binary data from the file
            with open(input_file_path, "rb") as file:
                data = file.read()

            # Calculate the number of rows needed for the image
            rows = (len(data) * 8) // img_width + 1  # Each byte has 8 bits

            # Create a blank (white) image with padding
            img_height = rows + 2 * padding
            img = Image.new("1", (img_width + 2 * padding, img_height), 1)  # '1' mode for binary (1-bit) image

            # Draw the SoftStrip pattern on the image
            for i, byte in enumerate(data):
                for bit in range(8):
                    if byte & (1 << (7 - bit)):  # Check each bit (MSB to LSB)
                        # Calculate pixel position
                        x = padding + (i * 8 + bit) % img_width
                        y = padding + (i * 8 + bit) // img_width
                        img.putpixel((x, y), 0)  # Set pixel to black (0) if bit is 1

            # Add alignment markers
            img = add_alignment_markers(img)

            # Save the generated SoftStrip image
            img.save(output_image_path)
            print(f"SoftStrip image saved to {output_image_path}")


        def add_alignment_markers(img):
            """
            Adds alignment markers (black and white alternating bars) at the top and bottom of the image.

            :param img: PIL Image object.
            :return: Image with alignment markers added.
            """
            img_width, img_height = img.size
            marker_height = 10  # Height of the alignment marker bars

            # Add top alignment markers
            for x in range(img_width):
                if x % 2 == 0:
                    for y in range(marker_height):
                        img.putpixel((x, y), 0)  # Black pixel for alignment

            # Add bottom alignment markers
            for x in range(img_width):
                if x % 2 == 0:
                    for y in range(img_height - marker_height, img_height):
                        img.putpixel((x, y), 0)  # Black pixel for alignment

            return img


        # Example usage
        input_file_path = input("path/to/your/input_file >>>")
        output_image_path = "softstrip.png"
        encode_softstrip(input_file_path, output_image_path)

    elif choice == "exit":
        exit(0)
