"""Draw the vision test image (tests/data/vision_test.png): a red circle, a blue square, a green triangle and the
word "INKLING" on white, 600x400 (not a multiple of 40 wide/high on purpose: exercises the padding path).
Usage: python tools/make_test_image.py [out.png]"""
import sys

from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1] if len(sys.argv) > 1 else "tests/data/vision_test.png"
img = Image.new("RGB", (610, 390), "white")
d = ImageDraw.Draw(img)
d.ellipse((40, 40, 200, 200), fill=(220, 30, 30))
d.rectangle((240, 50, 390, 200), fill=(30, 60, 220))
d.polygon([(500, 40), (430, 200), (570, 200)], fill=(30, 170, 60))
try:
    font = ImageFont.truetype("arial.ttf", 90)
except OSError:
    font = ImageFont.load_default()
d.text((120, 250), "INKLING", fill="black", font=font)
img.save(out)
print(out, img.size)
