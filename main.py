"""
Сервис предназначен для транскрибации аудиосообщений

Данильченко Вадим
"""


from flask import Flask, Response, request
import json, io, os, random

from transcribe_chunk_onnx import Wave2Vec2ONNXInference as wav2vec

path_audio = './audio'

app = Flask(__name__)

asr = wav2vec("./model/", './model/wav2vec2-large-xlsr-53-russian.onnx')

@app.route('/get_response', methods=['POST'])
def transcribe_from_local():
    jsreq = request.get_json(silent=True, force=True)


    filenames = os.listdir(path_audio) # "./audio/Моя запись 10.wav"
    fileio = os.path.join(path_audio, filenames[random.randint(0, len(filenames)-1)])

    if 'command' in jsreq:
        if 'transcribe' in jsreq['command']:
            if isinstance(jsreq['command'], dict):
                fileio = io.BytesIO(jsreq['command']['transcribe'].encode('latin-1'))
            data = asr.file_to_text(fileio)
        else:
            data = 'unknown command'
    else:
        data = 'unknown action'
    return Response(response=json.dumps({'data':data}), status=200, content_type='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8087)