import os
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import shutil
import uvicorn
from typing import List

from params import params
from run_model import create_model, process_data, run_model

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = create_model(params)

FILE_PREFIX = "./inputs/"
os.makedirs(FILE_PREFIX, exist_ok=True)

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=1239)

@app.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    # first file flair
    # second file t1
    for f in files:
        print(f.filename)
        print()
    folder_name = str(uuid.uuid4())
    prefix = os.path.join(FILE_PREFIX, folder_name + '/raw/sub_tmp')
    os.makedirs(prefix, exist_ok=True)
    file_location = os.path.join(prefix, "flair.nii.gz")
    with open(file_location, "wb") as f:
        shutil.copyfileobj(files[1].file, f)
    file_location = os.path.join(prefix, "t1.nii.gz")
    with open(file_location, "wb") as f:
        shutil.copyfileobj(files[0].file, f)
    print(f"files '{files[0].filename}' and '{files[1].filename}' uploaded")

    print(f"start processing data")
    params['data_dir'] = './inputs/' + folder_name
    params['base_dir'] = './outputs/' + folder_name
    params_processed = process_data(params)
    print(f"start running model")
    outputs = run_model(model, params_processed)
    print(outputs)

    file_path = outputs[0]
    if os.path.exists(file_path):
        return  {"status":"found", "url":"http://127.0.0.1:1239/files/t1_reg_seg.nii.gz"}
    else:
        return {"error": "file not found"}



@app.get("/files/{filename}")
async def get_file(filename: str):
    # Возвращаем файл для скачивания
    prefix = "./outputs/tmp/MS_DSA_NET_ps128_fs16/sub_tmp/"
    file_path = os.path.join(prefix, filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename)
    return {"error": "file not found"}
