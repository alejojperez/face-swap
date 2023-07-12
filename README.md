# Installation
- You must install ffmpeg, ideally using brew if on a Mac
- Create a python environment
```bash
python3 -m venv .
```
- Enter the new environment
```bash
source ./bin/activate
```
- Update pip and install all the requirements
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
- Go to this google drive link and dowload to the porjects root folder the file "inswapper_128.onnx"
https://drive.google.com/drive/folders/1luanq28v1k6l3sHj24jQbl4UvY_zOWwP?usp=sharing

# Usage

- Execute the main script and follow the instructions
```bash
python main.py
```