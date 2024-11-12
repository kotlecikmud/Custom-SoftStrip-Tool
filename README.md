# Custom-SoftStrip-Tool

The attached Python code performs the Softstrip encoder and decoder functions by using a header specifying the length of
the data. It uses the Pillow (PIL) libraries for image handling, NumPy for numerical calculations, and Tkinter to create
a simple graphical interface. This implementation is a proprietary solution intended for private use and combines
encoding and decoding functionalities in one tool.

**Softstrip Encoder**

The code works as follows to encode a text file into a Softstrip image:

* **Step 1:** It calculates the data length and adds it as a 16-bit binary header.
* **Step 2:** It converts the data to a binary stream, combining the length header with the binary representation of the
  data.
* **Step 3:** It encodes the bits into dibits, mapping pairs of bits to grayscale levels representing the Softstrip
  pattern.
    * "00" is white.
    * "01" is light gray.
    * "10" is dark gray.
    * "11" is black.
* **Step 4:** It sets the image width to 200 pixels and calculates the height to accommodate all the dibits.
* **Step 5:** It creates a new grayscale image ("L" mode) with the calculated dimensions, filled with white color.
* **Step 6:** It populates the image with dibits, setting the color of each pixel to the corresponding grayscale level.

**Softstrip Decoder**

To decode a Softstrip image back to the data:

* **Step 1:** It opens the image and converts it to grayscale.
* **Step 2:** It maps the grayscale levels back to dibits and then to a binary stream.
* **Step 3:** It extracts the 16-bit length header from the binary stream and converts it to an integer.
* **Step 4:** It converts the rest of the binary stream (excluding the header) to bytes.
* **Step 5:** It saves the decoded data to an output file.

**Graphical User Interface**

The code includes a simple graphical user interface (GUI) built using Tkinter that allows the user to:

* Select a text file to encode into a Softstrip image.
* Select a Softstrip image to decode into a text file.

The GUI displays success messages upon encoding and decoding.

**Usage**

The user can run the Python script and use the GUI to select a text file to encode or a Softstrip image to decode.
Encoded images are saved with a "_enc.png" suffix, and decoded text files with a "_dec.txt" suffix in the same directory
as the original file.

Mam nadzieję, że to tłumaczenie jest dla Ciebie pomocne! 

