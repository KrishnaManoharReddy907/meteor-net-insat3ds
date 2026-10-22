import pypdf
import os

pdf_path = r'docs/references/Self-Supervised_Temporal_Super-Resolution_Framework_With_METEOR-Net_for_Enhancing_Convective_Process_Reconstruction_in_Geostationary_Satellite_Cloud_Data (1).pdf'
out_dir = r'reports/paper_extracted'
os.makedirs(out_dir, exist_ok=True)

reader = pypdf.PdfReader(pdf_path)

all_text = []
for i, page in enumerate(reader.pages):
    page_num = i + 1
    text = page.extract_text() or ''
    out_page = os.path.join(out_dir, f'page_{page_num:02d}.txt')
    with open(out_page, 'w', encoding='utf-8') as f:
        f.write(f"=== PAGE {page_num} ===\n\n" + text)
    all_text.append(f"\n\n{'='*40}\n=== PAGE {page_num} ===\n{'='*40}\n\n" + text)

full_path = os.path.join(out_dir, 'full_paper_text.txt')
with open(full_path, 'w', encoding='utf-8') as f:
    f.writelines(all_text)

print(f"Extracted {len(reader.pages)} pages to {out_dir}")
