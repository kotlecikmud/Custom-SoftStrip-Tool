import os
import argparse
from PIL import Image, ImageDraw
import cv2
import numpy as np
from datetime import datetime
import traceback
import tkinter as tk


class StripEncoder:
    DATA_CHUNK_SIZE = 1024 * 2  # 2KB

    def __init__(self, input_file_path, output_image_path, dpi=300, module_size=10, data_columns=48):
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        self.input_file_path = input_file_path
        self.output_image_path = output_image_path
        self.dpi = dpi
        self.module_size = module_size
        self.data_columns = data_columns
        self.padding_modules = 2
        self.h_sync_bar_height = 2

    def encode(self):
        with open(self.input_file_path, "rb") as f:
            file_data = f.read()

        if not file_data:
            return

        filename = os.path.basename(self.input_file_path)
        original_file_size = len(file_data)

        chunks = [file_data[i:i + self.DATA_CHUNK_SIZE] for i in range(0, len(file_data), self.DATA_CHUNK_SIZE)]
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_size = len(chunk)
            padded_chunk = chunk + b'\x00' * (self.DATA_CHUNK_SIZE - len(chunk))

            payload = self._build_payload(padded_chunk, filename, i + 1, total_chunks, original_file_size, chunk_size)
            num_rows = (len(payload) * 8 + self.data_columns - 1) // self.data_columns
            image_width, image_height = self._calculate_dimensions(num_rows)

            img = Image.new("L", (image_width, image_height), 255)
            draw = ImageDraw.Draw(img)

            padding_pixels = self.module_size * self.padding_modules
            gray_background_color = 128
            draw.rectangle(
                [padding_pixels, padding_pixels, image_width - padding_pixels, image_height - padding_pixels],
                fill=gray_background_color
            )

            self._draw_data(draw, payload)

            base, ext = os.path.splitext(self.output_image_path)
            output_filename = f"{base}_{i + 1:02d}{ext}"

            img.save(output_filename, dpi=(self.dpi, self.dpi))

    def _calculate_checksum(self, data):
        return (256 - (sum(data) & 0xFF)) & 0xFF

    def _build_payload(self, data_chunk, filename, chunk_index, total_chunks, total_file_size, original_chunk_size):
        file_mod_time = int(os.path.getmtime(self.input_file_path))
        file_header = bytearray(b'\x02\x03')
        file_header.extend(total_chunks.to_bytes(2, 'little'))
        file_header.extend(chunk_index.to_bytes(2, 'little'))
        file_header.extend(total_file_size.to_bytes(4, 'little'))
        file_header.extend(original_chunk_size.to_bytes(2, 'little'))
        file_header.extend(filename.encode('ascii', 'replace'))
        file_header.extend(b'\x00')
        file_header.extend(file_mod_time.to_bytes(8, 'little', signed=False))
        strip_header = bytearray(b'PYCZIN\x01\x00\x00\x00\x14\x01')
        data_to_checksum = strip_header + file_header + data_chunk
        checksum = self._calculate_checksum(data_to_checksum)
        final_payload = bytearray()
        total_len = 1 + len(data_to_checksum)
        final_payload.extend(total_len.to_bytes(2, 'little'))
        final_payload.append(checksum)
        final_payload.extend(data_to_checksum)
        return bytes(final_payload)

    def _calculate_dimensions(self, num_rows):
        padding_pixels = self.module_size * self.padding_modules
        image_width = (self.data_columns + 2) * self.module_size + (2 * padding_pixels)
        image_height = (num_rows + self.h_sync_bar_height) * self.module_size + (2 * padding_pixels)
        return image_width, image_height

    def _draw_data(self, draw, payload):
        start_offset = self.module_size * self.padding_modules
        payload_rows = (len(payload) * 8 + self.data_columns - 1) // self.data_columns
        total_rows = payload_rows + self.h_sync_bar_height
        gray_background_color = 128

        # Draw vertical sync markers for the entire height
        for r in range(total_rows):
            y = start_offset + r * self.module_size
            color = 0 if r % 2 == 0 else gray_background_color

            # Left marker
            left_x = start_offset
            draw.rectangle((left_x, y, left_x + self.module_size - 1, y + self.module_size - 1), fill=color)

            # Right marker
            right_x = start_offset + (self.data_columns + 1) * self.module_size
            draw.rectangle((right_x, y, right_x + self.module_size - 1, y + self.module_size - 1), fill=color)

        # Draw horizontal sync bar (between the vertical markers)
        for r in range(self.h_sync_bar_height):
            for c in range(self.data_columns):
                color = 0 if (r + c) % 2 == 0 else gray_background_color
                x = start_offset + (c + 1) * self.module_size
                y = start_offset + r * self.module_size
                draw.rectangle((x, y, x + self.module_size - 1, y + self.module_size - 1), fill=color)

        # Draw payload data (below the horizontal sync bar)
        bit_index = 0
        for byte in payload:
            for i in range(8):
                if (byte >> (7 - i)) & 1:
                    row, col = divmod(bit_index, self.data_columns)
                    x = start_offset + (col + 1) * self.module_size
                    y = start_offset + (row + self.h_sync_bar_height) * self.module_size
                    draw.rectangle((x, y, x + self.module_size - 1, y + self.module_size - 1), fill=0)
                bit_index += 1


class StripDecoder:
    def __init__(self, input_image_path, data_columns=48):
        self.input_image_path = input_image_path
        self.data_columns = data_columns
        self.prepared_frame_width = 1000
        self.h_sync_bar_height = 2

    def decode_chunk(self, manual_corners=None):
        gray_image = self._preprocess_image()
        if gray_image is None: return None

        if manual_corners is not None:
            corners = manual_corners
        else:
            corners = self._find_data_grid_corners(gray_image)
            if corners is None: return None

        prepared_frame, M_inv = self._warp_to_prepared_frame(gray_image, corners)
        if prepared_frame is None or M_inv is None: return None

        bits = self._extract_bits_from_frame(prepared_frame)
        if bits is None or len(bits) < 16: return None

        len_bits = bits[:16]
        len_byte_str = "".join(map(str, len_bits))
        len_bytes = bytearray([int(len_byte_str[i:i + 8], 2) for i in range(0, len(len_byte_str), 8)])
        total_len = int.from_bytes(len_bytes, 'little')

        required_bits = (total_len + 2) * 8
        if len(bits) < required_bits: return None

        payload_bits = bits[:required_bits]

        payload = bytearray()
        for i in range(0, len(payload_bits), 8):
            byte_str = "".join(map(str, payload_bits[i:i + 8]))
            payload.append(int(byte_str, 2))

        try:
            if int.from_bytes(payload[0:2], 'little') != total_len: return None
            checksum_from_strip = payload[2]
            data_to_checksum = payload[3: 2 + total_len]
            calculated_checksum = self._calculate_checksum(data_to_checksum)
            if calculated_checksum != checksum_from_strip: return None

            cursor = 12
            cursor += 2
            total_chunks = int.from_bytes(data_to_checksum[cursor:cursor + 2], 'little')
            cursor += 2
            chunk_index = int.from_bytes(data_to_checksum[cursor:cursor + 2], 'little')
            cursor += 2
            total_file_size = int.from_bytes(data_to_checksum[cursor:cursor + 4], 'little')
            cursor += 4
            chunk_size = int.from_bytes(data_to_checksum[cursor:cursor + 2], 'little')
            cursor += 2
            filename_end = data_to_checksum.find(b'\x00', cursor)
            decoded_filename = data_to_checksum[cursor:filename_end].decode('ascii')
            cursor = filename_end + 1
            file_mod_time = int.from_bytes(data_to_checksum[cursor: cursor + 8], 'little', signed=False)
            decoded_date = datetime.fromtimestamp(file_mod_time)
            cursor += 8
            data_chunk = data_to_checksum[cursor:]
            header_info = {
                "total_chunks": total_chunks, "chunk_index": chunk_index, "total_file_size": total_file_size,
                "filename": decoded_filename, "date": decoded_date, "chunk_size": chunk_size
            }
            return data_chunk[:chunk_size], header_info
        except Exception:
            return None

    def _preprocess_image(self):
        return cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)

    def _find_data_grid_corners(self, gray_image):
        _, thresh_img = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        corners = np.zeros((4, 2), dtype="float32")
        s = box.sum(axis=1)
        corners[0] = box[np.argmin(s)]
        corners[2] = box[np.argmax(s)]
        diff = np.diff(box, axis=1)
        corners[1] = box[np.argmin(diff)]
        corners[3] = box[np.argmax(diff)]
        return corners

    def _warp_to_prepared_frame(self, image, corners):
        corners = np.float32(corners)
        width = np.linalg.norm(corners[0] - corners[1])
        height = np.linalg.norm(corners[0] - corners[3])
        if width == 0 or height == 0: return None, None

        dst_height = int(self.prepared_frame_width * (height / width))

        dst_points = np.array(
            [[0, 0], [self.prepared_frame_width - 1, 0], [self.prepared_frame_width - 1, dst_height - 1],
             [0, dst_height - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        inverse_matrix = cv2.getPerspectiveTransform(dst_points, corners)
        warped = cv2.warpPerspective(image, matrix, (self.prepared_frame_width, dst_height))
        return warped, inverse_matrix

    def _extract_bits_from_frame(self, frame):
        height, width = frame.shape
        module_size = width / (self.data_columns + 2)
        if module_size == 0: return None

        num_total_rows = int(height / module_size + 0.5)
        num_payload_rows = num_total_rows - self.h_sync_bar_height

        bits = []
        # Loop over potential payload rows, with a safety margin
        for r in range(num_payload_rows + 20):
            image_row = r + self.h_sync_bar_height
            for c in range(self.data_columns):
                center_x = int(((c + 1) + 0.5) * module_size)
                center_y = int((image_row + 0.5) * module_size)

                if 0 <= center_y < height and 0 <= center_x < width:
                    bits.append(1 if frame[center_y, center_x] < 64 else 0)
                else:
                    bits.append(0)
        return bits

    def _calculate_checksum(self, data):
        return (256 - (sum(data) & 0xFF)) & 0xFF


import sys

try:
    from gui import SoftstripGUI
except (ImportError, tk.TclError):
    SoftstripGUI = None


def main():
    parser = argparse.ArgumentParser(description="Cauzin Softstrip Encoder/Decoder")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    parser_encode = subparsers.add_parser("encode", help="Encode a file into a Softstrip image.")
    parser_encode.add_argument("input", help="Path to the file to encode.")
    parser_encode.add_argument("output", help="Path to save the output strip image.")
    subparsers.add_parser("gui", help="Launch the manual decoder GUI.")
    subparsers.add_parser("debug-multi-decode", help="Test the full multi-strip encode/decode pipeline.")

    if len(sys.argv) <= 1:
        args = parser.parse_args(["gui"])
    else:
        args = parser.parse_args()

    if args.command == "gui":
        if SoftstripGUI:
            app = SoftstripGUI()
            app.mainloop()
        else:
            print("GUI cannot be started. Ensure tkinter is installed and a display is available.")
    elif args.command == "encode":
        encoder = StripEncoder(args.input, args.output)
        encoder.encode()
    elif args.command == "debug-multi-decode":
        test_multi_decode()


def test_multi_decode():
    print("--- Running Multi-Strip Encode/Decode Test ---")
    large_file = "large_test_data.bin"
    output_base = "large_test_strip"
    output_ext = ".png"
    try:
        print("\n0. Creating large test file...")
        with open(large_file, "wb") as f:
            f.write(os.urandom(3000))
        print("\n1. Encoding large file...")
        encoder = StripEncoder(large_file, f"{output_base}{output_ext}")
        encoder.encode()
        generated_files = sorted([f for f in os.listdir() if f.startswith(output_base) and f.endswith(output_ext)])
        print(f"\n2. Found {len(generated_files)} strip images to decode.")
        print("\n3. Decoding strips...")
        all_chunks = {}
        header_info = None
        for f in generated_files:
            decoder = StripDecoder(f)
            result = decoder.decode_chunk()
            if result:
                chunk_data, chunk_header = result
                all_chunks[chunk_header['chunk_index']] = chunk_data
                if header_info is None or chunk_header['chunk_index'] == 1:
                    header_info = chunk_header
            else:
                raise RuntimeError(f"Decoding failed for {f}")
        print("\n4. Reassembling and verifying...")
        if not header_info or len(all_chunks) != header_info['total_chunks']:
            raise RuntimeError(
                f"Decoded {len(all_chunks)} chunks, but expected {header_info['total_chunks'] if header_info else 'N/A'}.")
        full_data = bytearray()
        for i in range(1, header_info['total_chunks'] + 1):
            full_data.extend(all_chunks[i])
        final_data = full_data[:header_info['total_file_size']]
        with open(large_file, "rb") as f:
            original_data = f.read()
        if final_data == original_data:
            print("\nSUCCESS: Reassembled data matches original file content.")
        else:
            print("\nFAILURE: Reassembled data does not match original file content.")
    except Exception as e:
        print(f"\n--- TEST FAILED: {e} ---")
        traceback.print_exc()
    finally:
        print("\n5. Cleaning up test files...")
        if os.path.exists(large_file):
            os.remove(large_file)
        generated_files = sorted([f for f in os.listdir() if f.startswith(output_base) and f.endswith(output_ext)])
        for f in generated_files:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    main()
