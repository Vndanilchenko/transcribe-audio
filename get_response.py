import requests, json

def req_func(data):
    res = requests.post('http://127.0.0.1:8087/get_response', data=json.dumps(data))
    text = res.json()
    print(text['data'])

# возьмёт произвольный файл из папки audio
data = {'command':'transcribe'}
req_func(data)

# передадим конкретный файл
filename = "./audio/Моя запись 10.wav"
with open(filename, 'rb') as f:
    # fileio = io.BytesIO(f.read())
    fileio = f.read()

    data = {'command':{'transcribe':fileio.decode('latin-1')}}
    req_func(data)

