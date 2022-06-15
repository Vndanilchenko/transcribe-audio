from onnxruntime.quantization.quantize import quantize
from transformers import Wav2Vec2ForCTC
import torch
import argparse

# took that script from: https://github.com/ccoreilly/wav2vec2-service/blob/master/convert_torch_to_onnx.py

def convert_to_onnx(model_id_or_path, onnx_model_name):
    print(f"Converting {model_id_or_path} to onnx")
    model = Wav2Vec2ForCTC.from_pretrained(model_id_or_path)
    # audio_len = 250000
    audio_len = 500000

    x = torch.randn(1, audio_len, requires_grad=True)

    torch.onnx.export(model,                        # model being run
                    x,                              # model input (or a tuple for multiple inputs)
                    onnx_model_name,                # where to save the model (can be a file or file-like object)
                    export_params=True,             # store the trained parameter weights inside the model file
                    opset_version=11,               # the ONNX version to export the model to
                    do_constant_folding=True,       # whether to execute constant folding for optimization
                    input_names = ['input'],        # the model's input names
                    output_names = ['output'],      # the model's output names
                    dynamic_axes={'input' : {1 : 'audio_len'},    # variable length axes
                                'output' : {1 : 'audio_len'}})

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    print("Starting quantization...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QUInt8)

    print(f"Quantized model saved to: {quantized_model_path}")

if __name__ == "__main__":
    convert = True
    quantize = False
    model_id_or_path = './model/'
    onnx_model_name = './model/wav2vec2-large-xlsr-53-russian_500k.onnx'

    if convert:
        convert_to_onnx(model_id_or_path, onnx_model_name)

    if quantize:
        quantized_model_name = './model/wav2vec2-large-xlsr-53-russian.quant.onnx'
        quantize_onnx_model(onnx_model_name, quantized_model_name)