from PIL import Image
import json


def show_image(image_array):
    img = Image.new("RGB", (28, 28), "white")
    for i, color in zip(range(len(image_array)), image_array):
        img.putpixel((i % 28, i // 28), tuple([255 - int(color)] * 3))
    img.show()
