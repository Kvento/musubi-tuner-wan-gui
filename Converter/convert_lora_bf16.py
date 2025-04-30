import torch
from safetensors.torch import load_file, save_file
import argparse

def convert_lora_to_bf16(input_path, output_path):
    # Проверяем поддержку BF16
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        print("Warning: Your GPU might not support bfloat16 natively!")
    
    # Загружаем исходную LoRA
    if input_path.endswith('.safetensors'):
        state_dict = load_file(input_path)
    else:  # Для .ckpt и других форматов
        state_dict = torch.load(input_path)
    
    # Конвертируем веса в BF16
    converted = {k: v.to(dtype=torch.bfloat16) for k, v in state_dict.items()}
    
    # Сохраняем результат
    if output_path.endswith('.safetensors'):
        save_file(converted, output_path)
    else:
        torch.save(converted, output_path)
    
    print(f"Successfully converted {input_path} to bfloat16 format: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input LoRA file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    args = parser.parse_args()
    
    convert_lora_to_bf16(args.input, args.output)