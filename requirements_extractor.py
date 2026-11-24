import argparse
import os
import re
from pathlib import Path
from typing import Literal

import ollama
import pandas as pd
import pymupdf

from pydantic import BaseModel
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table


# --------------------------------------------------
# 1. PYDANTIC OUTPUT SCHEMA
# --------------------------------------------------

class RequirementClassification(BaseModel):
    index: int
    requirement_type: Literal[
        "Performance",
        "Environmental",
        "Safety",
        "Interface"
    ]


class ClassificationResponse(BaseModel):
    classifications: list[RequirementClassification]


# --------------------------------------------------
# 2. CLASSIFY REQUIREMENTS USING OLLAMA
# --------------------------------------------------

def classify_requirements(requirements):

    requirement_list = []

    for index, requirement in enumerate(requirements, start=1):
        requirement_list.append({
            "index": index,
            "requirement_text": requirement["requirement_text"],
            "source_label": requirement["source_label"]
        })

    prompt = f"""
    Classify each engineering requirement into exactly one category.

    Categories:

    Performance:
    Speed, capacity, mass, endurance, service life,
    functional performance and EMC performance.

    Environmental:
    Temperature, vibration, shock and environmental protection.

    Safety:
    Fault handling, safety interlocks and prevention
    of unsafe operation.

    Interface:
    Power inputs, signals, connectors, communication
    and external inputs or outputs.

    Use the source label as supporting context, but classify
    based on the actual requirement meaning.

    Return one classification for every input index.
    Do not omit or invent indices.

    Requirements:
    {requirement_list}
    """

    for attempt in range(2):

        try:
            response = ollama.chat(
                model="qwen3:4b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                format=ClassificationResponse.model_json_schema(),
                options={"temperature": 0}
            )

            result = ClassificationResponse.model_validate_json(
                response.message.content
            )

            # Validate that all indexes are present exactly once.
            received_indexes = [
                item.index for item in result.classifications
            ]

            expected_indexes = list(range(1, len(requirements) + 1))

            if (
                len(received_indexes) != len(expected_indexes)
                or sorted(received_indexes) != expected_indexes
            ):
                raise ValueError(
                    "Ollama returned missing, duplicate or invalid indexes."
                )

            return result

        except Exception as error:

            print(
                f"Classification attempt {attempt + 1} failed: {error}"
            )

            if attempt == 1:

                output_dir = Path("requirements_output")
                output_dir.mkdir(parents=True, exist_ok=True)

                with open(
                    output_dir / "classification_errors.log",
                    "a",
                    encoding="utf-8"
                ) as log_file:
                    log_file.write(f"{error}\n")

                raise RuntimeError(
                    "Classification failed after one retry. "
                    "See classification_errors.log."
                )


# --------------------------------------------------
# 3. SPLIT SHALL STATEMENTS
# --------------------------------------------------

def split_shall_statements(text):

    # Correct missing spaces such as "shallwithstand".
    text = re.sub(
        r"shall(?=[a-z])",
        "shall ",
        text,
        flags=re.IGNORECASE
    )

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z])",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if re.search(r"\bshall\b", sentence, re.IGNORECASE)
    ]


# --------------------------------------------------
# 4. CREATE REQUIREMENT RECORDS
# --------------------------------------------------

def get_requirement_record(text, section=None, page=None):

    id_match = re.search(r"\b[A-Z]{2,}-\d{3}\b", text)

    label_match = re.search(
        r"\[([^\]]+)\]",
        text
    )

    return {
        "requirement_id": id_match.group(0) if id_match else None,
        "requirement_text": text,
        "source_label": label_match.group(1) if label_match else None,
        "section": section,
        "page": page
    }


# --------------------------------------------------
# 5. IDENTIFY SECTION HEADINGS
# --------------------------------------------------

def is_section_heading(paragraph):

    style_name = paragraph.style.name.lower()

    if "heading" in style_name:
        return True

    text = paragraph.text.strip()

    return bool(
        re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", text)
    )


# --------------------------------------------------
# 6. EXTRACT REQUIREMENTS FROM DOCX
# --------------------------------------------------

def extract_docx(file_path):

    document = Document(file_path)

    requirements = []
    current_section = None

    for element in document.element.body.iterchildren():

        if element.tag.endswith("}p"):

            paragraph = Paragraph(element, document)
            text = paragraph.text.strip()

            if not text:
                continue

            if is_section_heading(paragraph):
                current_section = text
                continue

            sentences = split_shall_statements(text)

            for sentence in sentences:

                requirements.append(
                    get_requirement_record(
                        sentence,
                        section=current_section,
                        page=None
                    )
                )

        elif element.tag.endswith("}tbl"):

            table = Table(element, document)

            for row in table.rows:

                cells = list(dict.fromkeys(
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ))

                row_text = " | ".join(cells)

                sentences = split_shall_statements(row_text)

                for sentence in sentences:

                    requirements.append(
                        get_requirement_record(
                            sentence,
                            section=current_section,
                            page=None
                        )
                    )

    return requirements


# --------------------------------------------------
# 7. EXTRACT REQUIREMENTS FROM PDF
# --------------------------------------------------

def extract_pdf(file_path):

    requirements = []
    current_section = None

    with pymupdf.open(file_path) as document:

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")

            for line in text.splitlines():

                line = line.strip()

                if re.match(
                    r"^\d+(?:\.\d+)*\s+[A-Z]",
                    line
                ):
                    current_section = line

            sentences = split_shall_statements(text)

            for sentence in sentences:

                requirements.append(
                    get_requirement_record(
                        sentence,
                        section=current_section,
                        page=page_number
                    )
                )

    return requirements


# --------------------------------------------------
# 8. SELECT DOCUMENT TYPE AND EXTRACT
# --------------------------------------------------

def extract_shall_statements(file_path):

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".docx":
        return extract_docx(file_path)

    elif extension == ".pdf":
        return extract_pdf(file_path)

    else:
        raise ValueError(
            "Only PDF and DOCX files are supported."
        )


# --------------------------------------------------
# 9. SAVE REQUIREMENTS TO CSV AND EXCEL
# --------------------------------------------------

def save_requirements(requirements, input_path):

    output_dir = Path("requirements_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = {
        "Requirement ID": "requirement_id",
        "Requirement Text": "requirement_text",
        "Requirement Type": "requirement_type",
        "Section": "section",
        "Page": "page"
    }

    if any(
        not item.get("requirement_type")
        for item in requirements
    ):
        raise ValueError(
            "Some requirements have no classification."
        )

    df = pd.DataFrame([
        {
            output_column: requirement.get(source_key)
            for output_column, source_key in columns.items()
        }
        for requirement in requirements
    ])

    file_name = Path(input_path).stem + "_requirements"

    csv_path = output_dir / f"{file_name}.csv"
    excel_path = output_dir / f"{file_name}.xlsx"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    df.to_excel(
        excel_path,
        index=False
    )

    print("\nFiles generated successfully!")
    print("CSV:", csv_path)
    print("Excel:", excel_path)
    print("Total requirements exported:", len(df))


# --------------------------------------------------
# 10. MAIN WORKFLOW
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Extract and classify SHALL statements."
    )

    parser.add_argument(
        "file_path",
        help="Path to the PDF or DOCX specification."
    )

    args = parser.parse_args()

    # Extract requirements.
    requirements = extract_shall_statements(args.file_path)

    if not requirements:
        print("No SHALL statements found.")
        return

    print("\nDocument:", os.path.basename(args.file_path))
    print("Total SHALL statements:", len(requirements))

    # Classify requirements.
    print("\nClassifying requirements with Ollama...")

    classification_result = classify_requirements(requirements)

    classifications = {
        item.index: item.requirement_type
        for item in classification_result.classifications
    }

    # Attach classifications to original dictionaries.
    for index, requirement in enumerate(requirements, start=1):

        requirement["requirement_type"] = classifications[index]

    # Display classifications.
    print("\nCLASSIFICATION RESULTS")

    for index, requirement in enumerate(requirements, start=1):

        print(
            index,
            requirement["requirement_id"],
            requirement["requirement_type"],
            requirement["requirement_text"]
        )

    # Export results.
    save_requirements(requirements, args.file_path)


# --------------------------------------------------
# 11. SCRIPT ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
