import fitz
pdfs = [
    r'm:\Projects\AI_Video\AI_Video-20260116T154526Z-3-001\AI_Video\AI Agent Model Stack Evaluation.pdf',
    r'm:\Projects\AI_Video\AI_Video-20260116T154526Z-3-001\AI_Video\AI Video Pipeline Architecture Review.pdf',
    r'm:\Projects\AI_Video\AI_Video-20260116T154526Z-3-001\AI_Video\Edge AI Agent Integration Guide.pdf'
]
with open('output_pdfs.txt', 'w', encoding='utf-8') as f:
    for pdf in pdfs:
        f.write(f'\n--- {pdf} ---\n')
        doc = fitz.open(pdf)
        for page in doc:
            f.write(page.get_text())
