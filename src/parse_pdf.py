import os
import pdfplumber

DATA_DIR = "data"
OUTPUT_DIR = "data"

def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text.strip()

def process_all_pdfs():
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            bank_code = filename.split("_")[0]
            pdf_path = os.path.join(DATA_DIR, filename)
            txt_output_path = os.path.join(OUTPUT_DIR, f"{bank_code}_annual_2023.txt")

            print(f"Processing {filename}...")
            extracted_text = extract_text_from_pdf(pdf_path)

            with open(txt_output_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            print(f"Saved extracted text to {txt_output_path}")

if __name__ == "__main__":
    process_all_pdfs()
