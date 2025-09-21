@"
# OCR-BOT

Screenshots your friend's lottery page and OCRs the Pick2/3/4/5 numbers.
Runs at :11 and :41 past the hour (10:00–22:11) via a tiny scheduler.

## Setup (Windows)
```powershell
py -m pip install -r requirements.txt
py -m playwright install
# Install Tesseract (and ensure it's on PATH)
#   https://github.com/UB-Mannheim/tesseract/wiki
tesseract --version
