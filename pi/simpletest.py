import numpy as np
from PIL import Image, ImageDraw
import adafruit_blinka_raspberry_pi5_piomatter as piomatter

width = 64
height = 64

geometry = piomatter.Geometry(
    width=width,
    height=height,
    n_addr_lines=5,
    rotation=piomatter.Orientation.Normal
)

canvas = Image.new('RGB', (width, height), (0, 0, 0))
draw = ImageDraw.Draw(canvas)
framebuffer = np.asarray(canvas) + 0

matrix = piomatter.PioMatter(
    colorspace=piomatter.Colorspace.RGB888Packed,
    pinout=piomatter.Pinout.AdafruitMatrixBonnet,
    framebuffer=framebuffer,
    geometry=geometry
)

draw.rectangle((2, 2, 30, 30), fill=0xFF0000)
draw.rectangle((34, 34, 62, 62), fill=0x0000FF)
framebuffer[:] = np.asarray(canvas)
matrix.show()
input("Press enter to exit")
