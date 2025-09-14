import os
import argparse
from PIL import Image, ImageDraw
import cv2
import numpy as np


# temp comment
class StripEncoder:
    def __init__(self, input_file_path, output_image_path, dpi=300, module_size=10, data_columns=48):
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        self.input_file_path = input_file_path
        self.output_image_path = output_image_path
        self.dpi = dpi
        self.module_size = module_size
        self.data_columns = data_columns
        self.x_padding_modules = 8  # Increased padding for finder patterns
        self.y_padding_modules = 8  # Increased padding for finder patterns

    def encode(self):
        with open(self.input_file_path, "rb") as f:
            file_data = f.read()
        filename = os.path.basename(self.input_file_path)
        payload = self._build_payload(file_data, filename)
        image_width, image_height = self._calculate_dimensions(len(payload))

        # Create an 8-bit grayscale image, default white background
        img = Image.new("L", (image_width, image_height), 255)
        draw = ImageDraw.Draw(img)

        # Draw the gray background for the data area
        x_start_pixels = self.x_padding_modules * self.module_size
        y_start_pixels = self.y_padding_modules * self.module_size
        data_width_pixels = self.data_columns * self.module_size
        num_rows = (len(payload) * 8 + self.data_columns - 1) // self.data_columns
        data_height_pixels = num_rows * self.module_size

        gray_background_color = 128
        draw.rectangle(
            [x_start_pixels, y_start_pixels, x_start_pixels + data_width_pixels, y_start_pixels + data_height_pixels],
            fill=gray_background_color
        )

        self._draw_alignment_markers(draw, image_width, image_height)
        self._draw_data(draw, payload)
        img.save(self.output_image_path, dpi=(self.dpi, self.dpi))
        print(f"Successfully encoded '{self.input_file_path}' to '{self.output_image_path}'")

    def _calculate_checksum(self, data):
        return (256 - (sum(data) & 0xFF)) & 0xFF

    def _build_payload(self, file_data, filename):
        file_header = bytearray(b'\x02\x01')
        file_header.extend(len(file_data).to_bytes(3, 'little'))
        file_header.extend(filename.encode('ascii'))
        file_header.extend(b'\x00\x00')
        strip_header = bytearray(b'PYCZIN\x01\x00\x00\x00\x14\x01')
        data_to_checksum = strip_header + file_header + file_data
        checksum = self._calculate_checksum(data_to_checksum)
        final_payload = bytearray()
        total_len = 1 + len(data_to_checksum)
        final_payload.extend(total_len.to_bytes(2, 'little'))
        final_payload.append(checksum)
        final_payload.extend(data_to_checksum)
        return bytes(final_payload)

    def _calculate_dimensions(self, payload_length_bytes):
        num_bits = payload_length_bytes * 8
        num_rows = (num_bits + self.data_columns - 1) // self.data_columns
        x_padding_pixels = self.module_size * self.x_padding_modules
        y_padding_pixels = self.module_size * self.y_padding_modules
        image_width = self.data_columns * self.module_size + (2 * x_padding_pixels)
        image_height = num_rows * self.module_size + (2 * y_padding_pixels)
        return image_width, image_height

    def _draw_finder_pattern(self, draw, top_left_x, top_left_y):
        ms = self.module_size
        # Outer black square (7x7)
        draw.rectangle([top_left_x, top_left_y, top_left_x + 7 * ms, top_left_y + 7 * ms], fill=0)
        # Middle white square (5x5)
        draw.rectangle([top_left_x + ms, top_left_y + ms, top_left_x + 6 * ms, top_left_y + 6 * ms], fill=255)
        # Inner black square (3x3)
        draw.rectangle([top_left_x + 2 * ms, top_left_y + 2 * ms, top_left_x + 5 * ms, top_left_y + 5 * ms], fill=0)

    def _draw_alignment_markers(self, draw, width, height):
        quiet_zone = self.module_size
        # Top-left finder
        self._draw_finder_pattern(draw, quiet_zone, quiet_zone)
        # Top-right finder
        self._draw_finder_pattern(draw, width - quiet_zone - 7 * self.module_size, quiet_zone)
        # Bottom-left finder
        self._draw_finder_pattern(draw, quiet_zone, height - quiet_zone - 7 * self.module_size)

    def _draw_data(self, draw, payload):
        x_start = self.module_size * self.x_padding_modules
        y_start = self.module_size * self.y_padding_modules
        bit_index = 0
        for byte in payload:
            for i in range(8):
                if (byte >> (7 - i)) & 1:
                    row, col = divmod(bit_index, self.data_columns)
                    x = x_start + col * self.module_size
                    y = y_start + row * self.module_size
                    draw.rectangle((x, y, x + self.module_size - 1, y + self.module_size - 1), fill=0)
                bit_index += 1


class StripDecoder:
    def __init__(self, input_image_path, output_file_path, data_columns=48):
        self.input_image_path = input_image_path
        self.output_file_path = output_file_path
        self.data_columns = data_columns
        self.prepared_frame_width = 1000

    def decode(self):
        # This is the main entry point for automatic decoding.
        result = self.find_and_get_corners_automatically()
        if result is None:
            return False

        gray_image, corners = result

        prepared_frame = self._warp_to_prepared_frame(gray_image, corners)
        if prepared_frame is None: return False
        payload = self._extract_data_from_frame(prepared_frame)
        if payload is None or len(payload) < 20: return False
        return self._verify_and_save(payload)

    def find_and_get_corners_automatically(self):
        gray_image = self._preprocess_image()
        if gray_image is None:
            return None

        corners = self._find_data_grid_corners(gray_image)

        if corners is None:
            print("Error: Could not find data grid automatically.")
            return None

        # Return the grayscale image and the 4 corners
        return gray_image, corners

    def decode_with_4_points(self, corners_list, update_callback=None):
        if len(corners_list) != 4:
            print("Error: Manual decoding requires exactly 4 points.")
            return False

        # Preprocess the image to get the grayscale version for warping
        gray_image = self._preprocess_image()
        if gray_image is None:
            return False

        # The corners from the GUI are the 4 corners of the data grid
        # The GUI ensures they are in TL, TR, BR, BL order
        corners = np.array(corners_list, dtype="float32")

        prepared_frame, M_inv = self._warp_to_prepared_frame(gray_image, corners)
        if prepared_frame is None: return False
        payload = self._extract_data_from_frame(prepared_frame, M_inv, update_callback)
        if payload is None or len(payload) < 20: return False
        return self._verify_and_save(payload)

    def get_debug_mask(self):
        gray_image = self._preprocess_image()
        if gray_image is None:
            return None

        lower_gray = 120
        upper_gray = 135
        mask = cv2.inRange(gray_image, lower_gray, upper_gray)
        kernel = np.ones((15, 15), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        return closed_mask

    def _preprocess_image(self):
        # Read the image in grayscale directly
        img = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img

    def _find_data_grid_corners(self, gray_image):
        # Isolate the gray data area background
        # The background is 128, so we create a tight mask around it.
        lower_gray = 120
        upper_gray = 135
        mask = cv2.inRange(gray_image, lower_gray, upper_gray)

        # Use morphological closing to fill in the holes from the black data bits
        kernel = np.ones((15, 15), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)

        # Find contours of the gray area
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("Error: Could not find the gray data grid.")
            return None

        # Find the largest contour, which should be the data grid
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the corners of the minimum area rectangle enclosing the contour
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)

        # Sort the box points to be in TL, TR, BR, BL order
        # Sum (x+y) is smallest at TL and largest at BR
        s = box.sum(axis=1)
        tl_index = np.argmin(s)

        # Reorder the corners based on the top-left index
        # The order from boxPoints can be arbitrary
        corners = np.roll(box, -tl_index, axis=0)

        # Ensure the first point is indeed the top-left and adjust if rotated
        # This logic is from the old implementation and is still useful
        if corners[1][1] > corners[3][1]:  # If TR y > BL y
            # It means the box is rotated by 90 degrees
            # The order would be TL, BL, BR, TR
            # Let's re-sort them
            tl = corners[0]
            bl = corners[1]
            br = corners[2]
            tr = corners[3]
            corners = np.array([tl, tr, br, bl], dtype="float32")

        return corners

    def _warp_to_prepared_frame(self, image, corners):
        # The corners are already sorted: tl, tr, br, bl
        # We need to ensure the corners are in the correct format for linalg.norm
        corners = np.float32(corners)

        # Calculate aspect ratio from source corners
        width = np.linalg.norm(corners[0] - corners[1])
        height = np.linalg.norm(corners[0] - corners[3])

        if width == 0 or height == 0:
            return None  # Avoid division by zero

        dst_height = int(self.prepared_frame_width * (height / width))

        dst_points = np.array([
            [0, 0],
            [self.prepared_frame_width - 1, 0],
            [self.prepared_frame_width - 1, dst_height - 1],
            [0, dst_height - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        inverse_matrix = cv2.getPerspectiveTransform(dst_points, corners)
        warped = cv2.warpPerspective(image, matrix, (self.prepared_frame_width, dst_height))
        # Return the warped image and the inverse transform matrix
        return warped, inverse_matrix

    def _extract_data_from_frame(self, frame, M_inv, update_callback=None):
        height, width = frame.shape
        module_size = width / self.data_columns
        # Use round() instead of int() to avoid floating point precision errors
        num_rows = round(height / module_size)
        bits = []
        for r in range(num_rows):
            for c in range(self.data_columns):
                center_x = int((c + 0.5) * module_size)
                center_y = int((r + 0.5) * module_size)
                if 0 <= center_y < height and 0 <= center_x < width:
                    # Check the grayscale value. Black (<64) is a 1 bit.
                    # Gray background (~128) or white (~255) is a 0 bit.
                    bits.append(1 if frame[center_y, center_x] < 64 else 0)

                    if update_callback:
                        # Transform point back to original image coordinates for visualization
                        point_in_warped = np.array([[[center_x, center_y]]], dtype=np.float32)
                        point_in_original = cv2.perspectiveTransform(point_in_warped, M_inv)
                        update_callback(point_in_original[0][0])
        payload = bytearray()
        for i in range(0, len(bits), 8):
            if len(bits[i:i + 8]) == 8:
                payload.append(int("".join(map(str, bits[i:i + 8])), 2))
        return payload

    def _calculate_checksum(self, data):
        return (256 - (sum(data) & 0xFF)) & 0xFF

    def _verify_and_save(self, payload):
        try:
            total_len = int.from_bytes(payload[0:2], 'little')
            if total_len == 0 or len(payload) < total_len + 2: return False
            checksum_from_strip = payload[2]
            data_to_checksum = payload[3: 2 + total_len]
            if self._calculate_checksum(data_to_checksum) != checksum_from_strip:
                print("Error: Checksum mismatch.")
                return False
            cursor = 12
            file_len = int.from_bytes(data_to_checksum[cursor + 2:cursor + 5], 'little')
            cursor += 5
            filename_end = data_to_checksum.find(b'\x00', cursor)
            cursor = filename_end + 1
            misc_info_len = data_to_checksum[cursor]
            cursor += 1 + misc_info_len
            file_data = data_to_checksum[cursor: cursor + file_len]
            with open(self.output_file_path, "wb") as f:
                f.write(file_data)
            print(f"Successfully decoded and saved to '{self.output_file_path}'")
            return True
        except Exception:
            return False


import sys
from gui import SoftstripGUI


def main():
    parser = argparse.ArgumentParser(description="Cauzin Softstrip Encoder/Decoder")
    subparsers = parser.add_subparsers(dest="command")

    # Encoder command
    parser_encode = subparsers.add_parser("encode", help="Encode a file into a Softstrip image.")
    parser_encode.add_argument("input", help="Path to the file to encode.")
    parser_encode.add_argument("output", help="Path to save the output strip image.")

    # New argument to launch the GUI explicitly
    subparsers.add_parser("gui", help="Launch the manual decoder GUI.")

    # New argument to launch the GUI explicitly
    subparsers.add_parser("gui", help="Launch the manual decoder GUI.")

    # If no command is given, or 'gui' is the command, launch the GUI
    if len(sys.argv) <= 1 or (len(sys.argv) > 1 and sys.argv[1] == 'gui'):
        app = SoftstripGUI()
        app.mainloop()
        return

    args = parser.parse_args()
    if args.command == "encode":
        encoder = StripEncoder(args.input, args.output)
        encoder.encode()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
