import os
import argparse
from PIL import Image, ImageDraw
import cv2
import numpy as np


class StripEncoder:
    def __init__(self, input_file_path, output_image_path, dpi=300, module_size=10, data_columns=48):
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        self.input_file_path = input_file_path
        self.output_image_path = output_image_path
        self.dpi = dpi
        self.module_size = module_size
        self.data_columns = data_columns
        self.x_padding_modules = 6
        self.y_padding_modules = 4

    def encode(self):
        with open(self.input_file_path, "rb") as f: file_data = f.read()
        filename = os.path.basename(self.input_file_path)
        payload = self._build_payload(file_data, filename)
        image_width, image_height = self._calculate_dimensions(len(payload))
        img = Image.new("1", (image_width, image_height), 1)
        draw = ImageDraw.Draw(img)
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
        x_padding = self.module_size * self.x_padding_modules
        y_padding = self.module_size * self.y_padding_modules
        image_width = self.data_columns * self.module_size + (2 * x_padding)
        image_height = num_rows * self.module_size + (2 * y_padding) + (self.module_size * 5)
        return image_width, image_height

    def _draw_alignment_markers(self, draw, width, height):
        x_pad, y_pad = self.module_size * 2, self.module_size * 2
        circ_diam = self.module_size * 3
        draw.ellipse((x_pad, y_pad, x_pad + circ_diam, y_pad + circ_diam), fill=0)
        rect_w, rect_h = self.module_size * 4, self.module_size * 2
        draw.rectangle((x_pad, height - y_pad - rect_h, x_pad + rect_w, height - y_pad), fill=0)

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
        processed_image = self._preprocess_image()
        if processed_image is None: return False
        data_grid_corners = self._find_data_grid_corners(processed_image)
        if data_grid_corners is None:
            print("Error: Could not find data grid.")
            return False
        prepared_frame = self._warp_to_prepared_frame(processed_image, data_grid_corners)
        if prepared_frame is None: return False
        payload = self._extract_data_from_frame(prepared_frame)
        if payload is None or len(payload) < 20: return False
        return self._verify_and_save(payload)

    def _preprocess_image(self):
        img = cv2.imread(self.input_image_path)
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    def _find_data_grid_corners(self, image):
        # Erode the image to separate touching bits
        kernel = np.ones((2, 2), np.uint8)
        eroded_image = cv2.erode(image, kernel, iterations=1)

        contours, _ = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        est_module_area = (image.shape[1] / (self.data_columns + 12)) ** 2
        min_area, max_area = est_module_area * 0.3, est_module_area * 2.5
        data_bit_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                _, _, w, h = cv2.boundingRect(cnt)
                if 0.6 < w / float(h) < 1.4:
                    data_bit_contours.append(cnt)

        if len(data_bit_contours) < 50:  # Lowered threshold
            return None

        all_points = np.concatenate(data_bit_contours)
        rect = cv2.minAreaRect(all_points)
        box = cv2.boxPoints(rect)
        return box

    def _warp_to_prepared_frame(self, image, corners):
        rect = cv2.minAreaRect(corners)
        box = cv2.boxPoints(cv2.minAreaRect(corners))
        width = int(rect[1][0])
        height = int(rect[1][1])

        # Determine the top-left corner to handle rotation
        s = box.sum(axis=1)
        tl_index = np.argmin(s)

        # Rotate box points so tl is first
        corners = np.roll(box, -tl_index, axis=0)

        # Recalculate width/height based on potentially rotated rect
        if corners[1][1] > corners[3][1]:  # If tr y > bl y
            width, height = height, width

        dst_height = int(self.prepared_frame_width * (height / width))
        dst_points = np.array(
            [[0, 0], [self.prepared_frame_width - 1, 0], [self.prepared_frame_width - 1, dst_height - 1],
             [0, dst_height - 1]], dtype="float32")

        matrix = cv2.getPerspectiveTransform(np.float32(corners), dst_points)
        warped = cv2.warpPerspective(image, matrix, (self.prepared_frame_width, dst_height))
        _, warped_thresh = cv2.threshold(warped, 128, 255, cv2.THRESH_BINARY)
        return warped_thresh

    def _extract_data_from_frame(self, frame):
        height, width = frame.shape
        module_size = width / self.data_columns
        num_rows = int(height / module_size)
        bits = []
        for r in range(num_rows):
            for c in range(self.data_columns):
                center_x = int((c + 0.5) * module_size)
                center_y = int((r + 0.5) * module_size)
                if 0 <= center_y < height and 0 <= center_x < width:
                    bits.append(1 if frame[center_y, center_x] > 128 else 0)
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
            if self._calculate_checksum(data_to_checksum) != checksum_from_strip: return False
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


def main():
    parser = argparse.ArgumentParser(description="Cauzin Softstrip Encoder/Decoder")
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_encode = subparsers.add_parser("encode", help="Encode a file into a Softstrip image.")
    parser_encode.add_argument("input", help="Path to the file to encode.")
    parser_encode.add_argument("output", help="Path to save the output strip image.")
    parser_decode = subparsers.add_parser("decode", help="Decode a Softstrip image to a file.")
    parser_decode.add_argument("input", help="Path to the strip image to decode.")
    parser_decode.add_argument("output", help="Path to save the decoded file.")
    args = parser.parse_args()

    if args.command == "encode":
        encoder = StripEncoder(args.input, args.output)
        encoder.encode()
    elif args.command == "decode":
        decoder = StripDecoder(args.input, args.output)
        if decoder.decode():
            print("Decoding process completed successfully.")
        else:
            print("Decoding process failed.")


if __name__ == "__main__":
    main()
