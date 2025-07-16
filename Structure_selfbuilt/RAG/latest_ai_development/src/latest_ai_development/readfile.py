def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()