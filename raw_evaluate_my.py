import json
import json
import os
import re
import sys
import argparse
import datasets
import fire
import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from their_peft.src.peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    if 'llama' in args.model.lower() or 'mistral' in args.model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )  # fix zwq
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": 0}
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model


def set_logger(filename):
    # 创建一个logger
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    import getpass
    username = getpass.getuser()
    log_path = "/root/autodl-fs/test_logs/"
    if "slurm" in username:
        log_path = "../test_logs/"
    os.makedirs(log_path, exist_ok=True)
    autodl_fs_handler = logging.FileHandler(os.path.join(log_path, filename))
    autodl_fs_handler.setLevel(logging.INFO)

    # 再创建一个handler，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    autodl_fs_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(autodl_fs_handler)

    return logger


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def load_data(args) -> datasets.Dataset:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = args.dataset
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    data = datasets.load_from_disk(file_path)
    keys = ['test', 'valid', 'validation']
    if os.environ.get("TEST_ON_TRAIN", False):
        keys = ["train", ]
    for key in keys:
        if key in data:
            data = data[key]
            break
    return data.select(list(range(min(args.test_sample_num, len(data)))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--test_sample_num', type=int, default=300)
    parser.add_argument('--adapter', default="KV")
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--set_pll_0', action='store_true', default=False)
    parser.add_argument('--fix_wrong_special_token', action='store_true', default=False)
    parser.add_argument("--explosion", action="store_true", default=False)
    return parser.parse_args()


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def evaluate(
        instruction,
        model,
        tokenizer,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
):
    if isinstance(instruction, str):
        prompt = generate_prompt(instruction, input)
    elif isinstance(instruction, list):
        prompt = []
        for p in instruction:
            prompt.append(generate_prompt(p))
    else:
        print(instruction)
        raise NotImplementedError

    inputs = tokenizer(prompt, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=2.0,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    s = generation_output.sequences
    # assert torch.any(s == 2)
    output = tokenizer.batch_decode(s)
    output = [o.split("### Response:")[1].strip() for o in output]
    return output


def nq_v1(data, model, tokenizer):
    instructions = data.get('question')
    outputs = evaluate(instructions, model, tokenizer)
    choices = data.get('short_answers')
    for output, std_answers, std_long_ans, instruction in zip(outputs, choices, data["long_answers"], instructions):
        f_ans = None
        flag = 0
        for choice in std_answers:
            if choice in output:
                flag = 1
                f_ans = choice
                break
        new_data = {
            'output_pred': output,
            'question': instruction,
            'choices': std_answers,
            'long_answers': ''.join(std_long_ans),
            'model_ans': f_ans,
        }
        yield flag, f_ans, new_data


def nq_std(data, model, tokenizer):
    instructions = data.get('instruction')
    outputs = evaluate(instructions, model, tokenizer)
    choices = data.get('answer')
    for output, std_answers, instruction in zip(outputs, choices, instructions):
        f_ans = None
        flag = 0
        for choice in std_answers:
            if choice in output:
                flag = 1
                f_ans = choice
                break
        new_data = {
            'output_pred': output,
            'question': instruction,
            'choices': std_answers,
            'model_ans': f_ans,
        }
        yield flag, f_ans, new_data


def medmcqa(data, model, tokenizer):
    instructions = data.get("instruction")
    outputs_b = evaluate(instructions, model, tokenizer)
    std_answers = data.get("answer")
    for std_ans, output, instruction in zip(std_answers, outputs_b, instructions):
        flag = 0
        if re.search(f"option {std_ans}" + r"[. !]?", output):
            flag = 1
        # if std_ans in outputs:
        #     flag = 1
        new_data = {
            'output_pred': output,
            'question': instruction,
            'state': flag,
            'std_ans': std_ans,
        }
        yield flag, None, new_data


def pubmedqa(data, model, tokenizer):
    instructions = data.get("instruction")
    # if os.environ.get("DEBUG_KV", False):
    #     print(instructions)
    outputs_b = evaluate(instructions, model, tokenizer)
    std_answers = data.get("answer")
    for std_ans, output, instruction in zip(std_answers, outputs_b, instructions):
        flag = 0
        if re.search(f"my final answer is {std_ans}" + r"[. !]?", output):
            flag = 1
        # if std_ans in outputs:
        #     flag = 1
        new_data = {
            'output_pred': output,
            'question': instruction,
            'state': flag,
            'std_ans': std_ans,
        }
        yield flag, None, new_data


def mmlu(data, model, tokenizer):
    instructions = data.get("instruction")
    outputs_b = evaluate(instructions, model, tokenizer)
    std_answers = data.get("answer")
    for std_ans, output, instruction in zip(std_answers, outputs_b, instructions):
        flag = 0
        _ = f"My final choice is ({std_ans}|{std_ans.lower()})"
        if re.search(_, output):
            flag = 1
        # if std_ans in outputs:
        #     flag = 1
        new_data = {
            'output_pred': output,
            'question': instruction,
            'state': flag,
            'std_ans': std_ans,
        }
        yield flag, None, new_data


def squad(data, model, tokenizer):
    instructions = data.get("instruction")
    outputs_b = evaluate(instructions, model, tokenizer)
    std_answers = data.get("answer")
    for std_ans, output, instruction in zip(std_answers, outputs_b, instructions):
        flag = 0
        low_output = output.lower()
        for ans in std_ans:
            if ans.lower() in low_output:
                flag = 1
                break
        new_data = {
            'output_pred': output,
            'question': instruction,
            'state': flag,
            'std_ans': std_ans,
        }
        yield flag, None, new_data


def tool(data, model, tokenizer):
    instructions = data.get("instruction")
    outputs_b = evaluate(instructions, model, tokenizer)
    std_answers = data.get("answer")
    for std_ans, output, instruction in zip(std_answers, outputs_b, instructions):
        # cnt = 0
        # low_output = output.lower()
        # for ans in std_ans:
        #     ans = ','.join(ans.split(',')[:3]).lower()
        #     if ans in low_output:
        #         cnt += 1
        # flag = cnt / len(std_ans)

        std_ans = [','.join(ans.split(',')[:3]).lower() for ans in std_ans]
        low_output = output.replace("API MAIN INFO: ", '')
        low_output = low_output.split("\n\n")
        low_output = [','.join(api.split(',')[:3]).lower() for api in low_output]
        merged = set(low_output).union(set(std_ans))
        inter = set(low_output).intersection(set(std_ans))
        flag = len(inter) / len(merged)

        new_data = {
            'output_pred': output,
            'question': instruction,
            'state': flag,
            'std_ans': std_ans,
        }
        yield flag, None, new_data


def gsm8k(data, model, tokenizer):
    def get_float_answers(s: str):
        ss = s.split(',')
        ans = []
        for x in ss:
            try:
                x = float(x)
                ans.append(x)
            except ValueError:
                pass

        return ans

    instructions = data.get("instruction")
    outputs_b = evaluate(instructions, model, tokenizer)
    std_answers = data.get("answer")
    for std_ans, output, instruction in zip(std_answers, outputs_b, instructions):
        flag = 0
        #   #### 260\nThe answer is: 260
        #   #### 2\nThe answer is: 2
        model_ans = ""
        try:
            model_ans = get_float_answers(
                re.search("[0-9.]+\nThe answer is: ([0-9.,]+)", output).group(1)
            )
            # model_ans = get_float_answers(output.split("The answer is:")[1])
            if len(model_ans) == len(std_ans):
                flag = 1
                for a, b in zip(model_ans, std_ans):
                    if abs(a - b) > 1e-6:
                        flag = 0
        except AttributeError:
            pass

        new_data = {
            'output_pred': output,
            'question': instruction,
            'state': flag,
            'std_ans': std_ans,
            'model_ans': model_ans,
        }
        yield flag, (std_ans, model_ans), new_data


def main(
        _load_model=load_model,
        _parse_args=parse_args,
):
    args = _parse_args()
    last_folder = os.path.basename(os.path.normpath(args.lora_weights))
    if last_folder == "none":
        last_folder = os.path.basename(os.path.normpath(args.base_model))
    if "checkpoint" in last_folder:
        f, l = os.path.split(os.path.normpath(args.lora_weights))
        _, f = os.path.split(f)
        last_folder = f"{f}@{l}"

    logger = set_logger(f"{args.adapter}-{last_folder}-{os.path.split(args.dataset)[-1]}.log")
    save_file = f'experiment/{args.model}-{args.adapter}-{os.path.split(args.dataset)[-1]}-{last_folder}.json'
    create_dir('experiment/')
    temp_output = open(f"{last_folder}.temp_log", "w", encoding='utf-8')

    dataset = load_data(args)
    tokenizer, model = _load_model(args)
    if args.fix_wrong_special_token:
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    total = len(dataset)
    correct = 0
    output_data = []
    pbar = tqdm(total=total)
    bs = args.bs
    data_len = len(dataset)
    cnt = 0
    logger.info("start test")
    logger.info(str(model.config))
    for idx in range(data_len):
        if idx * bs >= data_len:
            break
        data = dataset[idx * bs: (idx + 1) * bs]

        # if os.environ.get("DEBUG_KV", False):
        #     print(data)

        run_1 = None
        if "all_nq" in args.dataset:
            run_1 = nq_std
        elif 'nq_v1' in args.dataset:
            run_1 = nq_v1
        elif "nq_for_test" in args.dataset:
            run_1 = nq_v1
        elif 'medmc' in args.dataset:
            run_1 = medmcqa
        elif 'pubmed' in args.dataset:
            run_1 = pubmedqa
        elif 'mmlu' in args.dataset:
            run_1 = mmlu
        elif 'squad' in args.dataset:
            run_1 = squad
        elif 'gsm8k' in args.dataset:
            run_1 = gsm8k
        elif 'math' in args.dataset:
            run_1 = gsm8k
        elif 'tool' in args.dataset:
            run_1 = tool
        elif 'trivia' in args.dataset:
            run_1 = squad
        else:
            raise NotImplementedError

        # if os.environ.get("DEBUG_KV", False):
        #     print("args.dataset = ", args.dataset)
        #     print("run_1 == nq_v1", run_1 == nq_v1)

        for _, f_ans, res in run_1(data, model, tokenizer):
            correct += _
            output_data.append(res)
            logger.info(f'test:{cnt + 1}/{total} | accuracy {correct}  {correct / (cnt + 1)}')
            if os.environ.get("DEBUG_KV", False):
                # logger.debug(f"{res['question']}")
                # logger.debug(f"{res['output_pred']}")
                # logger.debug(f"{res['std_ans']}")
                if hasattr(model, "debug_print_cache_cnt_for_moe"):
                    logger.debug(model.debug_print_cache_cnt_for_moe())

            print(f'test:{cnt + 1}/{total} | accuracy {correct}  {correct / (cnt + 1)}', file=temp_output)
            pbar.update(1)
            cnt += 1

    pbar.close()
    logger.info('test finished')
    with open(save_file, 'w') as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
