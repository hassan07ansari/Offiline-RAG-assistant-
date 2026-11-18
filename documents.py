import pymupdf
from docx import Document
import re
import os

def load_pdf(file_path:str):
    document = pymupdf.open(file_path)

    pages = []

    for page_number,page in enumerate(document,start=1):
        text = page.get_text()

        if not text.strip():
            continue

        pages.append({
            "text" : text,
            "metadata" : {
                "document_name" : document.name,
                "page_number" : page_number
            }
        })

    document.close()

    return pages


def load_docx(file_path: str):

    document = Document(file_path)

    content = []

    for element in document.iter_inner_content(): #preserving the order of paragraphs and tables

        if element.__class__.__name__ == "Paragraph":

            text = element.text.strip()

            if text:
                content.append(text)

        elif element.__class__.__name__ == "Table":

            table_text = []

            for row in element.rows:
                row_text = [
                    cell.text.strip() for cell in row.cells
                ]

                table_text.append(" | ".join(row_text))

            content.append(
                "\n[TABLE]\n" +
                "\n".join(table_text) +
                "\n[/TABLE]\n"
            )

    return [{
        "text": "\n\n".join(content),
        "metadata": {
            "document_name": file_path,
            "page_number": None,
            "file_type": "docx"
        }
    }]


def is_heading(line):

    line = line.strip()

    if not line:
        return False

    # Numbered headings: 1, 1.1, 2.3.4
    numbered = re.match(
        r"^\d+(?:\.\d+)*\.?\s+[A-Z]",
        line
    )

    # Uppercase headings
    uppercase = (
        len(line) < 120
        and line.isupper()
        and len(line.split()) <= 15
        and any(char.isalpha() for char in line)
    )

    return bool(numbered or uppercase)




def chunk_documents(pages, max_chunk_size=1800):

    chunks = []
    current_section = "General"
    buffer = []
    current_metadata = {}

    def save_chunk():

        if not buffer:
            return

        chunk_text = "\n".join(buffer).strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **current_metadata,
                    "section": current_section
                }
            })

        buffer.clear()

    for page in pages:

        text = page["text"]
        metadata = page["metadata"]

        current_metadata = metadata.copy()

        lines = text.splitlines()

        i = 0

        while i < len(lines):

            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Preserve complete tables
            if line == "[TABLE]":

                save_chunk()

                table_lines = []

                i += 1

                while i < len(lines) and lines[i].strip() != "[/TABLE]":
                    table_lines.append(lines[i])
                    i += 1

                table_text = "\n".join(table_lines)

                chunks.append({
                    "text": table_text,
                    "metadata": {
                        **current_metadata,
                        "section": current_section,
                        "content_type": "table"
                    }
                })

                i += 1
                continue

            # Detect a new section
            if is_heading(line):

                save_chunk()

                current_section = line
                buffer.append(line)

                i += 1
                continue

            # Add normal content
            current_size = sum(len(item) for item in buffer)

            if current_size + len(line) > max_chunk_size:
                save_chunk()
                buffer.append(current_section)

            buffer.append(line)

            i += 1

        save_chunk()

    return chunks

def load_document(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")
