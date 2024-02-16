import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# 监控的目录
monitor_path = '/root/autodl-fs/trained_models'
# JSON记录文件
json_log_file = 'operation_log.json'

# 操作记录字典
operation_log = {}

# 读取已有的操作记录
if os.path.isfile(json_log_file):
    with open(json_log_file, 'r') as f:
        operation_log = json.load(f)


def execute_bash_and_record(path):
    # 执行bash脚本
    sample_num = 501
    if "tool_v1" in path:
        sample_num = 451
    tasks = ["nq_v1", "metamathqa_v1", "pubmedqa_v2", "squad_v2", "tool_v1"]
    test_task = None
    for task in tasks:
        if task in path:
            test_task = task
    if test_task == "metamathqa_v1":
        test_task = "gsm8k_v1"
    weight = path
    base_model = 'linhvu/decapoda-research-llama-7b-hf' if "llama" in path.lower() else 'mistralai/Mistral-7B-v0.1'
    model_type = "llama" if "llama" in path.lower() else "mistral"
    sample_num = str(sample_num)

    if "checkpoint" in weight and "-kv-" not in weight:
        return

    try:
        if "-kv-" in path:
            subprocess.check_call(['bash', '/root/autodl-fs/kvxxx/LLM-Adapters/config/auto_eval_kv.sh',
                                   test_task, weight, sample_num, base_model, model_type])
        else:
            subprocess.check_call(['bash', '/root/autodl-fs/kvxxx/LLM-Adapters/config/auto_eval_baseline.sh',
                                   test_task, weight, sample_num, base_model, model_type])
        # 记录成功状态
        operation_log[path] = 'Success'
    except:
        # 记录失败状态
        operation_log[path] = 'Failed'
    # 保存操作记录到JSON文件
    with open(json_log_file, 'w') as f:
        json.dump(operation_log, f, indent=4)


def check_and_execute_initial_files():
    # 检查所有修改时间在两天以内的文件
    for root, dirs, files in os.walk(monitor_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if (re.fullmatch(r".+?/checkpoint-[0-9]+?", dir_path) or
                    re.fullmatch(r".+?/trained_models/[^/]+", dir_path)):

                # 获取文件最后修改时间
                mtime = datetime.fromtimestamp(os.path.getmtime(dir_path))
                # 检查文件是否在两天以内被修改
                if datetime.now() - timedelta(days=6) < mtime < datetime.now() - timedelta(days=1):
                    # 检查是否已执行过bash脚本或执行失败
                    if operation_log.get(dir_path) != 'Success':
                        execute_bash_and_record(dir_path)


class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # 只有在文件被修改时才触发
        if event.is_directory:
            dir_path = event.src_path
            if (re.fullmatch(r".+?/checkpoint-[0-9]+?", dir_path) or
                    re.fullmatch(r".+?/trained_models/[^/]+", dir_path)):
                # 确保文件已经更新完成
                time.sleep(180)  # 简单的延迟，等待文件写入完成
                # 执行bash脚本
                execute_bash_and_record(dir_path)


if __name__ == "__main__":
    # 首先检查所有修改时间在两天以内的文件
    check_and_execute_initial_files()

    # 设置事件处理器
    event_handler = MyHandler()
    # 设置监控对象
    observer = Observer()
    observer.schedule(event_handler, monitor_path, recursive=True)
    # 开始监控
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
