from PIL import Image, ImageDraw, ImageFont

W, H = 1100, 220
BG = (255, 255, 255)
DARK_BLUE = (27, 58, 92)
ORANGE = (255, 107, 53)

img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
draw.rounded_rectangle([0, 0, W, H], radius=20, fill=BG)

try:
    font_hockey = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 110)
    font_roi = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 110)
except:
    font_hockey = ImageFont.load_default()
    font_roi = ImageFont.load_default()

hockey_text = "HOCKEY"
bbox = draw.textbbox((0, 0), hockey_text, font=font_hockey)
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]
y = (H - th) // 2 - bbox[1]
draw.text((80, y), hockey_text, font=font_hockey, fill=DARK_BLUE)

div_x = 80 + tw + 30
draw.rectangle([div_x, 30, div_x + 4, H - 30], fill=ORANGE)

roi_text = "ROI"
bbox2 = draw.textbbox((0, 0), roi_text, font=font_roi)
draw.text((div_x + 30, y), roi_text, font=font_roi, fill=ORANGE)

img.save("/Users/ashgarg/Library/CloudStorage/OneDrive-Personal/NHL analysis/brand/hockeyroi_banner.png", "PNG")
print("Saved!")
