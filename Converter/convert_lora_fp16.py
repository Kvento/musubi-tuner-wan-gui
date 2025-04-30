import torch
from safetensors.torch import load_file, save_file
import argparse

def convert_lora_to_fp16(input_path, output_path):
    # Принудительно используем CPU
    device = torch.device('cpu')
    
    # Загружаем исходную LoRA
    if input_path.endswith('.safetensors'):
        state_dict = load_file(input_path, device=str(device))
    else:  # Для .ckpt и других форматов
        state_dict = torch.load(input_path, map_location=device)
    
    # Конвертируем веса в FP16
    converted = {k: v.to(dtype=torch.float16) for k, v in state_dict.items()}
    
    # Сохраняем результат
    if output_path.endswith('.safetensors'):
        save_file(converted, output_path)
    else:
        torch.save(converted, output_path)
    
    print(f"Successfully converted {input_path} to float16 format: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input LoRA file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    args = parser.parse_args()
    
    convert_lora_to_fp16(args.input, args.output)
