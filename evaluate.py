import os

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 构建命令行参数
command = [
    'python',
    'src/evaluate.py',
    '--model_name_or_path', '/home/featurize/Qwen-1_8B-Chat',
    '--template', 'vanilla',
    '--finetuning_type', 'lora',
    '--task', 'ques',
    '--split', 'test',
    '--lang', 'zh',
    '--n_shot', '5',
    '--batch_size', '4'
]

# 执行命令
os.system(' '.join(command))