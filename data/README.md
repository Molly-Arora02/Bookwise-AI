# BookWise AI — Source Data Directory

This directory holds document assets for BookWise AI.

## Supplied Document

- **Filename**: `History_2024-25.pdf`
- **Publisher**: Orient BlackSwan
- **Catalogue Title**: History 2024–25 Catalogue
- **Scope**: Academic and general history books, monographs, series, author indices, title indices, and publisher information.
- **Total Pages**: 91 pages
- **Format**: Adobe PDF 1.6 (Digitally rendered from Adobe InDesign 20.0 Windows)

## Privacy and Copyright Notice

To respect publishing copyrights and repository size limits, the raw source PDF (`data/*.pdf`) is excluded from public version control via `.gitignore`.

To use your local copy of the catalogue:
1. Place `History_2024-25.pdf` into this `data/` directory.
2. Run the ingestion command:
   ```bash
   python ingest.py
   ```
3. The ingestion pipeline will verify file integrity using SHA-256 and index all 91 pages into persistent ChromaDB vector storage.
