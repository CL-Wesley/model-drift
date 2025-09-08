from fastapi import APIRouter, UploadFile, File
from datetime import datetime
import uuid
import os
import pandas as pd

router = APIRouter(prefix="/api/v1/data-drift/upload", tags=["Data Drift - Upload"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def parse_csv(file_path: str):
    """Helper to parse CSV and extract metadata"""
    df = pd.read_csv(file_path)
    rows = len(df)
    columns = df.shape[1]
    column_names = df.columns.tolist()
    data_types = df.dtypes.apply(lambda x: x.name).to_dict()
    preview = df.head(5).to_dict(orient="records")
    return rows, columns, column_names, data_types, preview

@router.post("/reference")
async def upload_reference_dataset(file: UploadFile = File(...)):
    dataset_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, "latest_reference.csv")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Parse CSV
    rows, columns, column_names, data_types, preview = parse_csv(file_path)

    return {
        "status": "success",
        "message": "Reference dataset uploaded successfully",
        "data": {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": rows,
            "columns": columns,
            "column_names": column_names,
            "dataTypes": data_types,
            "preview": preview,
            "upload_time": datetime.utcnow().isoformat()
        }
    }

@router.post("/current")
async def upload_current_dataset(file: UploadFile = File(...)):
    dataset_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, "latest_current.csv")


    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Parse CSV
    rows, columns, column_names, data_types, preview = parse_csv(file_path)

    return {
        "status": "success",
        "message": "Current dataset uploaded successfully",
        "data": {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": rows,
            "columns": columns,
            "column_names": column_names,
            "dataTypes": data_types,
            "preview": preview,
            "upload_time": datetime.utcnow().isoformat()
        }
    }
