from pathlib import Path

pages_dir = Path("out/or_md_flash2/pages")

if not pages_dir.exists():
    print(f"Папка не найдена: {pages_dir}")
    exit()

query = "Железы — это органы"
found = 0

for f in sorted(pages_dir.glob("*.md")):
    text = f.read_text(encoding="utf-8")
    if query.lower() in text.lower():
        print(f"\n{'='*40}")
        print(f"Файл: {f.name}")
        print(f"{'='*40}")
        print(text[:500])
        found += 1

if found == 0:
    print(f"\nНичего не найдено по запросу '{query}' в папке {pages_dir}")