from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def process_page(pdf_page, vector_page, image_path_page, vector_pdf_page):
    save(pdf_page, vector_page, image_path_page, vector_pdf_page)
    to_sharp(image_path_page)

def save(pdf_page, vector_page, image_path_page, vector_pdf_page):
    pdf_page.save(image_path_page, "PNG")
    writer = PdfWriter()
    writer.add_page(vector_page)
    vector_pdf_page.parent.mkdir(parents=True, exist_ok=True)
    with open(vector_pdf_page, "wb") as f:
        writer.write(f)

def to_sharp(image_path_page):
    image = cv2.imread(str(image_path_page))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    clean = cv2.fastNlMeansDenoising(binary, h=30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    sharpened = cv2.dilate(clean, kernel, iterations=1)
    sharpened = cv2.erode(sharpened, kernel, iterations=1)

    output_path = Path(image_path_page)
    cv2.imwrite(output_path, sharpened)
    return sharpened

def preprocess(pdf_path, image_path="/tmp/floor_plan.png"):
    pages = convert_from_path(
        pdf_path,
        dpi=200,  # <--- FIX 1: Change this from 400 to 200
    )
    reader = PdfReader(pdf_path)
    image_path_pages = list()
    vector_pdf_pages = list()
    image_path = Path(image_path)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = list()
        for index, (pdf_page, vector_page) in enumerate(zip(pages, reader.pages)):
            image_path_page = image_path.parent.joinpath(image_path.stem).with_suffix(f".{str(index).zfill(2)}{image_path.suffix}")
            vector_pdf_page = image_path.parent.joinpath(str(index).zfill(2)).joinpath(f"scaled_{image_path.stem}").with_suffix(".pdf")
            futures.append(
                executor.submit(
                    process_page,
                    pdf_page,
                    vector_page,
                    image_path_page,
                    vector_pdf_page,
                )
            )
            image_path_pages.append(image_path_page)
            vector_pdf_pages.append(vector_pdf_page)
        [future.result() for future in futures]

    return vector_pdf_pages, image_path_pages
            image_path_pages.append(image_path_page)
            vector_pdf_pages.append(vector_pdf_page)
        [future.result() for future in futures]

    return vector_pdf_pages, image_path_pages
