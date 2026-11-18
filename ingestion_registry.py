import json
import os

REGISTRY_PATH = "data/ingestion_registry.json"


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {"documents": []}

    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def is_already_processed(file_hash, pipeline_version):

    registry = load_registry()

    for document in registry["documents"]:
        if (
            document.get("file_hash") == file_hash
            and document.get("pipeline_version") == pipeline_version
            and document.get("status") == "completed"
        ):
            return True

    return False


def update_registry(document_info):

    registry = load_registry()

    # Replace an existing record for the same file and pipeline
    registry["documents"] = [
        document
        for document in registry["documents"]
        if not (
            document.get("file_hash") == document_info["file_hash"]
            and document.get("pipeline_version") == document_info["pipeline_version"]
        )
    ]

    registry["documents"].append(document_info)

    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)

    temp_path = REGISTRY_PATH + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)

    os.replace(temp_path, REGISTRY_PATH)
