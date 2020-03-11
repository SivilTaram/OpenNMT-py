import requests

if __name__ == '__main__':
    file_url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz"
    r = requests.get(file_url, stream=True)
    with open("bert-base-chinese.tar.gz", "wb") as pdf:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                pdf.write(chunk)
