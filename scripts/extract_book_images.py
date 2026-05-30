"""
extract_book_images.py — Fase 1 do v2.0
Extrai imagens dos livros dermatológicos (Azulay 2015 + Glossário) usando PyMuPDF.
Gera data/base_nacional/images_raw/ + data/base_nacional/raw_extraction.csv.
"""
import sys
import csv
from pathlib import Path
import fitz  # PyMuPDF

sys.stdout.reconfigure(line_buffering=True)

ROOT      = Path(__file__).resolve().parent.parent
PDF_DIR   = ROOT / "docs" / "entregas" / "2.0_artigo_multimodal_base_nacional" / "material_base"
OUT_IMG   = ROOT / "data" / "base_nacional" / "images_raw"
OUT_CSV   = ROOT / "data" / "base_nacional" / "raw_extraction.csv"
OUT_IMG.mkdir(parents=True, exist_ok=True)

# Os PDFs sao descobertos por glob (lida com diferencas de normalizacao Unicode NFC/NFD)
def find_pdfs():
    books = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        name_lower = pdf.name.lower()
        if "azulay" in name_lower:
            books.append(("Azulay", pdf))
        elif "glosario" in name_lower or "glossario" in name_lower:
            books.append(("Glossario", pdf))
        else:
            label = pdf.stem[:20]
            books.append((label, pdf))
    return books

BOOKS = find_pdfs()

# Filtros mínimos na extração (não na curadoria — esses são só para evitar lixo extremo)
MIN_WIDTH  = 50
MIN_HEIGHT = 50

def get_nearby_text(page, img_bbox, max_chars=600):
    """Captura blocos de texto próximos à imagem (acima e abaixo)."""
    blocks = page.get_text("blocks")
    candidates = []
    for b in blocks:
        x0, y0, x1, y1, text, *_ = b
        if not text.strip():
            continue
        # bloco abaixo da imagem (legenda típica)
        if y0 >= img_bbox.y1 - 5 and y0 < img_bbox.y1 + 250:
            candidates.append((y0 - img_bbox.y1, text.strip()))
        # bloco acima
        elif y1 <= img_bbox.y0 + 5 and y1 > img_bbox.y0 - 250:
            candidates.append((img_bbox.y0 - y1, text.strip()))
    candidates.sort()
    combined = " | ".join(t for _, t in candidates)
    return combined[:max_chars]

def extract_caption(nearby_text):
    """Pega o primeiro fragmento que parece ser legenda (começa com Fig./Figura)."""
    if not nearby_text:
        return ""
    lower = nearby_text.lower()
    for marker in ("figura ", "fig. ", "fig ", "imagem ", "foto ", "quadro "):
        idx = lower.find(marker)
        if idx >= 0:
            return nearby_text[idx:idx + 300]
    # se nada, retorna primeiros 200 chars
    return nearby_text[:300]

def main():
    rows = []
    total_imgs = 0

    for source_label, pdf_path in BOOKS:
        if not pdf_path.exists():
            print(f"[ERRO] PDF não encontrado: {pdf_path}")
            continue

        # Pular Azulay se ja foi processado (idempotencia)
        if source_label == "Azulay" and (OUT_IMG / "Azulay_p0001_i00.png").exists():
            print(f"\n=== {source_label} ja processado, pulando ===")
            continue

        print(f"\n=== Processando {source_label} ({pdf_path.name}) ===")
        doc = fitz.open(pdf_path)
        n_pages = doc.page_count
        print(f"  Páginas: {n_pages}")

        book_imgs = 0
        for page_idx in range(n_pages):
            page = doc[page_idx]
            images = page.get_images(full=True)
            for img_i, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    # Skip imagens CMYK convertendo para RGB
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    w, h = pix.width, pix.height
                    if w < MIN_WIDTH or h < MIN_HEIGHT:
                        pix = None
                        continue

                    image_id = f"{source_label}_p{page_idx+1:04d}_i{img_i:02d}"
                    out_path = OUT_IMG / f"{image_id}.png"
                    pix.save(out_path)
                    pix = None

                    # bbox do primeiro rect da imagem na página
                    rects = page.get_image_rects(xref)
                    bbox = rects[0] if rects else fitz.Rect(0, 0, 0, 0)
                    nearby_text = get_nearby_text(page, bbox)
                    nearby_caption = extract_caption(nearby_text)

                    rows.append({
                        "image_id":       image_id,
                        "source_book":    source_label,
                        "source_page":    page_idx + 1,
                        "image_index":    img_i,
                        "width":          w,
                        "height":         h,
                        "nearby_caption": nearby_caption.replace("\n", " "),
                        "nearby_text":    nearby_text.replace("\n", " "),
                    })
                    book_imgs += 1
                except Exception as e:
                    print(f"    [warn] página {page_idx+1} img {img_i}: {e}")

            if (page_idx + 1) % 50 == 0:
                print(f"  página {page_idx+1}/{n_pages} | imagens extraídas até aqui: {book_imgs}", flush=True)

        doc.close()
        print(f"  {source_label} concluído: {book_imgs} imagens")
        total_imgs += book_imgs

    print(f"\n=== Total: {total_imgs} imagens em {OUT_IMG} ===")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "source_book", "source_page", "image_index",
            "width", "height", "nearby_caption", "nearby_text"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV salvo: {OUT_CSV}")

if __name__ == "__main__":
    main()
