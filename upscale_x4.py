from PIL import Image
import sys


def scale_image_4x(input_path, output_path):
    # Otwórz obraz
    img = Image.open(input_path)

    # Pobierz oryginalne wymiary obrazu
    original_width, original_height = img.size

    # Oblicz nowe wymiary (4x większe)
    new_width = original_width * 4
    new_height = original_height * 4

    # Skaluj obraz przy użyciu interpolacji najbliższego sąsiada
    scaled_img = img.resize((new_width, new_height), Image.NEAREST)

    # Zapisz przeskalowany obraz
    scaled_img.save(output_path)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    input_image_path = input("input image: ")
    scale_image_4x(input_image_path, "output.png")
