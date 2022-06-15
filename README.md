# Сервис предназначен для транскрибации аудиосообщений на русском языке 

### в основе модель wav2vec2: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian
### выполнена конвертация в формат ONNX, время работы уменьшилось более чем в 2 раза

    Reference (21сек)
    здрасте поступило предложение из банка вот хочу поподробнее узнать может быть есть какие-то 
    в какие-то условия более подробные может с договора вышлите очень интересно так как давно 
    сотрудничаю с вашим банком и новый продукт хотели бы использовать
    
    Prediction (7.1сек)
    здравстью поступило предложение из банку вот хочу по подробнее знать может быть есть какие-то 
    какие-то условия более подробные может договор вышлите очень интересно так как давно 
    сотрудничаем с вашим банком и новый продукты хотели бы использовать
    
    
### скрипты:
* основной - main.py
* модель - transcribe_chunk_onnx.py
* проверка - get_response.py
* ковертация в onnx - scripts/convert_torch_to_onnx.py

@author: Vadim Danilchenko

@email: vndanilchenko@gmail.com