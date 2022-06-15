"""
тест модели по транскрибации аудиозаписей на русском языке
"""


from asrecognition import ASREngine
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from codetiming import Timer

# asr = ASREngine("ru", model_path="jonatasgrosman/wav2vec2-large-xlsr-53-russian")
asr = ASREngine("ru", model_path="./model/")

path = './audio'
audio_paths = os.listdir(path)
i = 0

# Моя запись 10.wav
# 0:00:21.200213
# ЗДРАВСТЬЮ ПОСТУПИЛО ПРЕДЛОЖЕНИЕ ИЗ БАНКУ ВОТ ХОЧУ ПО ПОДРОБНЕЕ ЗНАТЬ МОЖЕТ БЫТЬ ЕСТЬ КАКИЕ-ТО КАКИЕ-ТО УСЛОВИЯ БОЛЕЕ ПОДРОБНЫЕ МОЖЕТ ДОГОВОР ВЫШЛИТЕ ОЧЕНЬ ИНТЕРЕСНО ТАК КАК ДАВНО СОТРУДНИЧАЕМ С ВАШИМ БАНКОМ И НОВЫЙ ПРОДУКТЫ ХОТЕЛИ БЫ ИСПОЛЬЗОВАТЬ

with Timer(text="\nTotal elapsed time: {:1f}"):
    for filename in tqdm(audio_paths):
        if filename[-3:] == 'wav':
            start_time = datetime.today()
            transcriptions = asr.transcribe([path + '/' + filename])
            end_time = datetime.today()
            transcrib_time = end_time - start_time
            print(filename)
            print(transcrib_time)
            print(transcriptions[0]['transcription'])
            break