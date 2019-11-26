from PIL import Image
import pytesseract
text = pytesseract.image_to_string(Image.open('C:\\Users\\Administrator\\Desktop\\test.png'),lang='chi_sim',config='--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"')
print(text)