__version__ = "00.01.01.00"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from softstrip import StripEncoder, StripDecoder


class SoftstripGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Softstrip Encoder/Decoder")
        self.geometry("1200x800")

        self.tabs_data = []  # To store data for each tab
        self.image_paths = []  # Keep a simple list of paths for the autodetect_all function

        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=300, relief=tk.SUNKEN, borderwidth=1)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        controls_label = tk.Label(left_frame, text="Controls", font=("Helvetica", 16, "bold"))
        controls_label.pack(pady=10)

        encode_frame = tk.LabelFrame(left_frame, text="Encode File", padx=10, pady=10)
        encode_frame.pack(pady=20, padx=10, fill=tk.X)
        self.encode_btn = tk.Button(encode_frame, text="Select File and Encode", command=self.encode_file)
        self.encode_btn.pack(pady=5, fill=tk.X)

        decode_frame = tk.LabelFrame(left_frame, text="Decode Strips", padx=10, pady=10)
        decode_frame.pack(pady=20, padx=10, fill=tk.X)

        self.load_btn = tk.Button(decode_frame, text="Load Strip Image(s)", command=self.load_images)
        self.load_btn.pack(pady=5, fill=tk.X)

        self.autodetect_btn = tk.Button(decode_frame, text="Autodetect & Decode All",
                                        command=self.autodetect_and_decode, state=tk.DISABLED)
        self.autodetect_btn.pack(pady=5, fill=tk.X)

        manual_frame = tk.LabelFrame(decode_frame, text="Manual Selection", padx=5, pady=5)
        manual_frame.pack(pady=10, fill=tk.X)

        self.manual_mode_var = tk.StringVar(value="bbox")
        self.bbox_radio = tk.Radiobutton(manual_frame, text="Bounding Box", variable=self.manual_mode_var, value="bbox",
                                         command=self.reset_manual_selection)
        self.bbox_radio.pack(anchor=tk.W)
        self.corners_radio = tk.Radiobutton(manual_frame, text="4 Corners", variable=self.manual_mode_var,
                                            value="corners", command=self.reset_manual_selection)
        self.corners_radio.pack(anchor=tk.W)

        self.manual_decode_btn = tk.Button(manual_frame, text="Decode Current Tab Manually",
                                           command=self.decode_with_manual_points, state=tk.DISABLED)
        self.manual_decode_btn.pack(pady=5, fill=tk.X)

        self.clear_btn = tk.Button(manual_frame, text="Clear Manual Selection", command=self.reset_manual_selection,
                                   state=tk.DISABLED)
        self.clear_btn.pack(pady=5, fill=tk.X)

        self.status_label = tk.Label(self, text="Load an image to start.", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.magnifier = Toplevel(self)
        self.magnifier.overrideredirect(True)
        self.magnifier.withdraw()
        self.mag_canvas = tk.Canvas(self.magnifier, width=150, height=150, borderwidth=2, relief=tk.RIDGE)
        self.mag_canvas.pack()
        self.mag_canvas.create_line(75, 0, 75, 150, fill="red")
        self.mag_canvas.create_line(0, 75, 150, 75, fill="red")
        self.mag_photo = None

    def get_current_tab_data(self):
        try:
            current_tab_index = self.notebook.index(self.notebook.select())
            if current_tab_index < len(self.tabs_data):
                return self.tabs_data[current_tab_index]
        except (tk.TclError, IndexError):
            return None
        return None

    def encode_file(self):
        input_path = filedialog.askopenfilename(title="Select File to Encode")
        if not input_path: return
        output_path = filedialog.asksaveasfilename(title="Save Strip Image As", defaultextension=".png",
                                                   filetypes=[("PNG files", "*.png")])
        if not output_path: return
        try:
            encoder = StripEncoder(input_path, output_path)
            encoder.encode()
            messagebox.showinfo("Success", f"File encoded successfully. Check the output folder for the strip images.")
        except Exception as e:
            messagebox.showerror("Encoding Error", f"An error occurred during encoding: {e}")

    def load_images(self):
        paths = filedialog.askopenfilenames(title="Select Strip Images",
                                            filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not paths: return
        self.image_paths = sorted(list(paths))
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)
        self.tabs_data.clear()
        for path in self.image_paths:
            self.add_image_tab(path)
        if self.image_paths:
            self.autodetect_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
            self.manual_decode_btn.config(state=tk.NORMAL)

    def add_image_tab(self, path):
        tab_frame = ttk.Frame(self.notebook)
        tab_name = os.path.basename(path)
        self.notebook.add(tab_frame, text=tab_name)
        canvas = tk.Canvas(tab_frame, bg="gray")
        canvas.pack(fill=tk.BOTH, expand=True)
        image = Image.open(path)
        tab_data = {
            "path": path, "image": image, "canvas": canvas, "photo": None,
            "scale_factor": 1.0, "manual_points": [], "rect_start": None, "rect_id": None
        }
        self.tabs_data.append(tab_data)
        canvas.bind("<Button-1>", self.on_canvas_click)
        canvas.bind("<B1-Motion>", self.on_canvas_drag)
        canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        canvas.bind("<Motion>", self.on_mouse_move)
        self.notebook.select(tab_frame)
        self.update_idletasks()
        self.display_image_in_canvas(tab_data)

    def display_image_in_canvas(self, tab_data):
        canvas, image = tab_data["canvas"], tab_data["image"]
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        img_w, img_h = image.size
        if canvas_w < 2 or canvas_h < 2:
            self.after(50, lambda: self.display_image_in_canvas(tab_data))
            return
        scale_factor = min(canvas_w / img_w, canvas_h / img_h)
        display_w, display_h = int(img_w * scale_factor), int(img_h * scale_factor)
        display_image = image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        tab_data["photo"] = photo
        tab_data["scale_factor"] = scale_factor

    def autodetect_and_decode(self):
        if not self.image_paths: return
        all_chunks, final_header = {}, None
        for i, path in enumerate(self.image_paths):
            self.notebook.select(i)
            self.update_idletasks()
            decoder = StripDecoder(path)
            result = decoder.decode_chunk()
            if not result:
                messagebox.showerror("Decoding Error",
                                     f"Failed to automatically decode {os.path.basename(path)}. Try manual selection.")
                return
            chunk_data, header_info = result
            all_chunks[header_info['chunk_index']] = chunk_data
            if final_header is None: final_header = header_info
        self.reassemble_and_save(all_chunks, final_header)

    def reassemble_and_save(self, all_chunks, header_info):
        if not header_info or len(all_chunks) != header_info['total_chunks']:
            messagebox.showerror("Reassembly Error",
                                 "Could not reassemble file: not all chunks were decoded successfully.")
            return
        full_data = bytearray()
        for i in range(1, header_info['total_chunks'] + 1):
            if i not in all_chunks:
                messagebox.showerror("Reassembly Error", f"Missing chunk #{i}!")
                return
            full_data.extend(all_chunks[i])
        final_data = full_data[:header_info['total_file_size']]
        output_path = filedialog.asksaveasfilename(initialfile=header_info['filename'], title="Save As",
                                                   filetypes=[("All files", "*.*")])
        if not output_path: return
        try:
            with open(output_path, "wb") as f:
                f.write(final_data)
            os.utime(output_path, (header_info['date'].timestamp(), header_info['date'].timestamp()))
            messagebox.showinfo("Success",
                                f"File '{os.path.basename(output_path)}' was successfully reassembled and saved.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save the file: {e}")

    def on_canvas_click(self, event):
        tab_data = self.get_current_tab_data()
        if not tab_data: return
        mode = self.manual_mode_var.get()
        if mode == "corners":
            if len(tab_data["manual_points"]) < 4:
                tab_data["manual_points"].append((event.x, event.y))
                tab_data["canvas"].create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red",
                                               outline="white", tags="manual_point")
                tab_data["canvas"].create_text(event.x, event.y - 10, text=f"#{len(tab_data['manual_points'])}",
                                               fill="white", tags="manual_point")
        elif mode == "bbox" and tab_data["rect_start"] is None:
            tab_data["rect_start"] = (event.x, event.y)
            tab_data["rect_id"] = tab_data["canvas"].create_rectangle(tab_data["rect_start"][0],
                                                                      tab_data["rect_start"][1],
                                                                      tab_data["rect_start"][0],
                                                                      tab_data["rect_start"][1], outline="cyan",
                                                                      width=2, tags="manual_point")

    def on_canvas_drag(self, event):
        tab_data = self.get_current_tab_data()
        if not tab_data or tab_data["rect_start"] is None: return
        tab_data["canvas"].coords(tab_data["rect_id"], tab_data["rect_start"][0], tab_data["rect_start"][1], event.x,
                                  event.y)

    def on_canvas_release(self, event):
        tab_data = self.get_current_tab_data()
        if not tab_data or tab_data["rect_start"] is None: return
        end_x, end_y = event.x, event.y
        start_x, start_y = tab_data["rect_start"]
        tab_data["rect_start"] = None
        tl = (min(start_x, end_x), min(start_y, end_y))
        tr = (max(start_x, end_x), min(start_y, end_y))
        br = (max(start_x, end_x), max(start_y, end_y))
        bl = (min(start_x, end_x), max(start_y, end_y))
        tab_data["manual_points"] = [tl, tr, br, bl]

    def decode_with_manual_points(self):
        tab_data = self.get_current_tab_data()
        if not tab_data or not tab_data["manual_points"]:
            messagebox.showerror("Error", "Please load an image and select points on the current tab first.")
            return
        original_corners = np.array(
            [(p[0] / tab_data["scale_factor"], p[1] / tab_data["scale_factor"]) for p in tab_data["manual_points"]],
            dtype="float32")
        decoder = StripDecoder(tab_data["path"])
        result = decoder.decode_chunk(manual_corners=original_corners)
        if not result:
            messagebox.showerror("Decoding Error",
                                 f"Failed to decode {os.path.basename(tab_data['path'])} with the selected points.")
            self.reset_manual_selection()
            return
        chunk_data, header_info = result
        self.reassemble_and_save({header_info['chunk_index']: chunk_data}, header_info)

    def reset_manual_selection(self):
        tab_data = self.get_current_tab_data()
        if not tab_data: return
        tab_data["manual_points"] = []
        tab_data["rect_start"] = None
        if tab_data.get("rect_id"):
            tab_data["canvas"].delete(tab_data["rect_id"])
        tab_data["canvas"].delete("manual_point")
        mode = self.manual_mode_var.get()
        self.status_label.config(text=f"Select {'bounding box' if mode == 'bbox' else '4 corners'} on the current tab.")

    def on_mouse_move(self, event):
        tab_data = self.get_current_tab_data()
        if not tab_data or not tab_data.get("image"):
            self.magnifier.withdraw()
            return
        canvas = event.widget
        x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        if 0 <= x < canvas.winfo_width() and 0 <= y < canvas.winfo_height():
            self.magnifier.geometry(f"+{self.winfo_x() + x + 20}+{self.winfo_y() + y + 20}")
            self.magnifier.deiconify()
            orig_x = int(x / tab_data["scale_factor"])
            orig_y = int(y / tab_data["scale_factor"])
            box_size = 25
            box = (orig_x - box_size, orig_y - box_size, orig_x + box_size, orig_y + box_size)
            region = tab_data["image"].crop(box)
            zoomed_region = region.resize((150, 150), Image.Resampling.NEAREST)
            self.mag_photo = ImageTk.PhotoImage(zoomed_region)
            self.mag_canvas.create_image(0, 0, anchor=tk.NW, image=self.mag_photo)
            self.mag_canvas.create_line(75, 0, 75, 150, fill="red")
            self.mag_canvas.create_line(0, 75, 150, 75, fill="red")
        else:
            self.magnifier.withdraw()


if __name__ == "__main__":
    app = SoftstripGUI()
    app.mainloop()
