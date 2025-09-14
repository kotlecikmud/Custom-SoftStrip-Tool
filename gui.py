import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import time

__version__ = "00.01.00.00"

class SoftstripGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Softstrip Manual Decoder")
        self.geometry("800x600")

        self.corners = []
        self.image_path = None
        self.original_image = None
        self.photo_image = None
        self.scale_factor = 1.0
        self.file_to_encode = None

        # Main frame
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)

        # Magnifier
        self.magnifier = None
        self.magnifier_canvas = None
        self.magnifier_photo = None
        self.magnifier_size = 100  # The size of the magnifier window
        self.magnifier_zoom = 3  # The zoom factor

        # Control frame
        control_frame = tk.Frame(main_frame, width=200, padx=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Instructions
        instructions = (
            "Instructions:\n"
            "1. Load Image.\n"
            "2. Use 'Autodetect' or manually click the 4 corners\n"
            "   of the gray data area in order:\n"
            "   - Top-Left\n"
            "   - Top-Right\n"
            "   - Bottom-Right\n"
            "   - Bottom-Left\n"
            "3. Click Decode."
        )
        self.instructions_label = tk.Label(control_frame, text=instructions, justify=tk.LEFT)
        self.instructions_label.pack(pady=10)

        # Buttons
        self.load_button = tk.Button(control_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(fill=tk.X, pady=5)

        self.decode_button = tk.Button(control_frame, text="Decode", command=self.decode_image)
        self.decode_button.pack(fill=tk.X, pady=5)

        self.autodetect_button = tk.Button(control_frame, text="Autodetect Corners", command=self.autodetect_corners)
        self.autodetect_button.pack(fill=tk.X, pady=5)

        self.reset_button = tk.Button(control_frame, text="Reset Clicks", command=self.reset_clicks)
        self.reset_button.pack(fill=tk.X, pady=5)

        # --- Debug Frame ---
        debug_frame = tk.LabelFrame(control_frame, text="Debug")
        debug_frame.pack(fill=tk.X, expand=False, pady=20)

        self.show_mask_var = tk.IntVar()
        self.show_mask_check = tk.Checkbutton(debug_frame, text="Show Debug Mask", variable=self.show_mask_var)
        self.show_mask_check.pack(fill=tk.X, padx=5, anchor="w")

        self.visualize_bits_var = tk.IntVar()
        self.visualize_bits_check = tk.Checkbutton(debug_frame, text="Visualize Bit Processing",
                                                   variable=self.visualize_bits_var)
        self.visualize_bits_check.pack(fill=tk.X, padx=5, anchor="w")

        # Status Label
        self.status_label = tk.Label(control_frame, text="Status: Ready", wraplength=180)
        self.status_label.pack(pady=20, side=tk.BOTTOM)

        # --- Encoder Frame ---
        encoder_frame = tk.LabelFrame(control_frame, text="Encoder")
        encoder_frame.pack(fill=tk.X, expand=False, pady=20)

        self.select_file_button = tk.Button(encoder_frame, text="Select File to Encode",
                                            command=self.select_file_to_encode)
        self.select_file_button.pack(fill=tk.X, pady=5, padx=5)

        self.encode_file_label = tk.Label(encoder_frame, text="No file selected.", wraplength=180)
        self.encode_file_label.pack(pady=5)

        self.encode_button = tk.Button(encoder_frame, text="Encode and Save As...", command=self.encode_file)
        self.encode_button.pack(fill=tk.X, pady=5, padx=5)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("PNG Images", "*.png"), ("All files", "*.*")]
        )
        if not self.image_path:
            return

        self.reset_clicks()
        self.original_image = Image.open(self.image_path)

        # Scale image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Make sure canvas has a size before scaling
        if canvas_width == 1 or canvas_height == 1:
            self.update()  # Update to get the real size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

        img_w, img_h = self.original_image.size

        self.scale_factor = 1.0
        if img_w > canvas_width or img_h > canvas_height:
            w_ratio = canvas_width / img_w
            h_ratio = canvas_height / img_h
            self.scale_factor = min(w_ratio, h_ratio)

        new_width = int(img_w * self.scale_factor)
        new_height = int(img_h * self.scale_factor)

        display_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.photo_image = ImageTk.PhotoImage(display_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.update_status(f"Loaded: {os.path.basename(self.image_path)}")

    def on_canvas_click(self, event):
        if not self.photo_image:
            self.update_status("Please load an image first.")
            return

        if len(self.corners) >= 4:
            self.update_status("Already have 4 points. Click Reset to start over.")
            return

        # Get click coordinates on the canvas
        canvas_x, canvas_y = event.x, event.y

        # Convert to original image coordinates
        original_x = canvas_x / self.scale_factor
        original_y = canvas_y / self.scale_factor

        self.corners.append((original_x, original_y))

        # Draw a circle and a number on the canvas at the click location
        marker_number = len(self.corners)
        self.canvas.create_oval(canvas_x - 3, canvas_y - 3, canvas_x + 3, canvas_y + 3, fill="red", outline="red",
                                tags="corner_marker")
        self.canvas.create_text(canvas_x + 10, canvas_y, text=f"#{marker_number}", fill="white", tags="corner_marker",
                                anchor="w")
        self.update_status(f"Point {len(self.corners)} selected. Need {3 - len(self.corners)} more.")

    def autodetect_corners(self):
        if not self.image_path:
            self.update_status("Please load an image first.")
            return

        self.update_status("Autodetecting corners...")
        self.reset_clicks()

        from softstrip_tool import StripDecoder
        decoder = StripDecoder(self.image_path, "")  # Output path not needed for corner detection

        if self.show_mask_var.get():
            mask = decoder.get_debug_mask()
            if mask is not None:
                mask_img = Image.fromarray(mask)
                mask_win = tk.Toplevel(self)
                mask_win.title("Debug Mask")
                mask_photo = ImageTk.PhotoImage(mask_img)
                mask_label = tk.Label(mask_win, image=mask_photo)
                mask_label.image = mask_photo  # Keep a reference
                mask_label.pack()

        result = decoder.find_and_get_corners_automatically()

        if result is None:
            self.update_status("Autodetection failed.")
            messagebox.showerror("Autodetection Failed", "Could not automatically find the finder patterns.")
            return

        # The method returns (gray_image, corners_array)
        _, corners_array = result

        # The corners are TL, TR, BR, BL.
        self.corners = corners_array.tolist()

        # Draw all 4 corners on the canvas
        for i, corner in enumerate(self.corners):
            scaled_corner = (corner[0] * self.scale_factor, corner[1] * self.scale_factor)
            self.canvas.create_oval(scaled_corner[0] - 3, scaled_corner[1] - 3, scaled_corner[0] + 3,
                                    scaled_corner[1] + 3, fill="blue", outline="blue", tags="corner_marker")
            self.canvas.create_text(scaled_corner[0] + 10, scaled_corner[1], text=f"#{i + 1}", fill="white",
                                    tags="corner_marker", anchor="w")

        self.update_status("Autodetection complete. Verify points or click Decode.")

    def decode_image(self):
        if len(self.corners) != 4:
            self.update_status(f"Error: Need 4 points to decode, but have {len(self.corners)}.")
            return

        self.update_status("Decoding...")

        from softstrip_tool import StripDecoder
        output_filename = self.image_path.rsplit('.', 1)[0] + "_decoded.txt"

        # Prepare the callback if visualization is enabled
        callback = None
        if self.visualize_bits_var.get():
            callback = self.update_bit_highlight

        try:
            decoder = StripDecoder(self.image_path, output_filename)
            # Pass the callback to the decoder
            success = decoder.decode_with_4_points(self.corners, update_callback=callback)

            # Clean up the final highlight marker
            self.canvas.delete("bit_highlight")

            if success:
                self.update_status(f"Success! Saved to {output_filename}")
                messagebox.showinfo("Success", f"Image decoded successfully!\n\nOutput saved to:\n{output_filename}")
            else:
                self.update_status("Decoding failed. Check console for errors.")
                messagebox.showerror("Failure",
                                     "Could not decode the image. The corners may be inaccurate or the image may be corrupt.")
        except Exception as e:
            self.update_status(f"An error occurred: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred during decoding:\n\n{e}")

    def update_bit_highlight(self, original_coords):
        # Delete the previous highlight
        self.canvas.delete("bit_highlight")

        # Scale the coordinate to the canvas
        canvas_x = original_coords[0] * self.scale_factor
        canvas_y = original_coords[1] * self.scale_factor

        # Draw a new highlight
        self.canvas.create_oval(canvas_x - 5, canvas_y - 5, canvas_x + 5, canvas_y + 5,
                                fill="red", outline="red", tags="bit_highlight")

        # Force canvas to update and pause briefly
        self.canvas.update_idletasks()
        time.sleep(0.001)

    def reset_clicks(self):
        self.corners = []
        self.canvas.delete("corner_marker")
        self.update_status("Clicks reset. Select 3 points.")

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def select_file_to_encode(self):
        self.file_to_encode = filedialog.askopenfilename()
        if self.file_to_encode:
            self.encode_file_label.config(text=f"Selected: {os.path.basename(self.file_to_encode)}")
            self.update_status("File selected for encoding. Click 'Encode and Save As...'")
        else:
            self.encode_file_label.config(text="No file selected.")

    def encode_file(self):
        if not self.file_to_encode:
            messagebox.showerror("Error", "Please select a file to encode first.")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Save Encoded Image As..."
        )

        if not output_path:
            return  # User cancelled the save dialog

        from softstrip_tool import StripEncoder
        try:
            self.update_status(f"Encoding to {os.path.basename(output_path)}...")
            encoder = StripEncoder(self.file_to_encode, output_path)
            encoder.encode()
            self.update_status("Encoding successful.")
            messagebox.showinfo("Success", f"File encoded successfully to:\n{output_path}")
        except Exception as e:
            self.update_status(f"Encoding failed: {e}")
            messagebox.showerror("Encoding Failed", f"An error occurred during encoding:\n\n{e}")

    def on_mouse_move(self, event):
        if not self.original_image:
            return

        if not self.magnifier:
            self.magnifier = tk.Toplevel(self)
            self.magnifier.overrideredirect(True)  # Remove window decorations
            self.magnifier.attributes("-topmost", True)  # Keep on top
            self.magnifier_canvas = tk.Canvas(self.magnifier, width=self.magnifier_size, height=self.magnifier_size)
            self.magnifier_canvas.pack()

        # Position the magnifier window near the cursor
        self.magnifier.geometry(f"+{event.x_root + 20}+{event.y_root + 20}")

        # Calculate the region to crop from the original image
        original_x = event.x / self.scale_factor
        original_y = event.y / self.scale_factor

        box_half_size = (self.magnifier_size / self.magnifier_zoom) / 2

        left = original_x - box_half_size
        top = original_y - box_half_size
        right = original_x + box_half_size
        bottom = original_y + box_half_size

        cropped_image = self.original_image.crop((left, top, right, bottom))

        # Resize (magnify) the cropped image
        magnified_image = cropped_image.resize((self.magnifier_size, self.magnifier_size), Image.Resampling.NEAREST)

        self.magnifier_photo = ImageTk.PhotoImage(magnified_image)
        self.magnifier_canvas.create_image(0, 0, anchor=tk.NW, image=self.magnifier_photo)

    def on_mouse_leave(self, event):
        if self.magnifier:
            self.magnifier.destroy()
            self.magnifier = None
            self.magnifier_canvas = None


if __name__ == "__main__":
    app = SoftstripGUI()
    app.mainloop()
