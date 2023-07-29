"""
  This script provides an exmaple to wrap TencentPretrain for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os, random
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--prediction_path", type=str, default="output.ceval")

    args = parser.parse_args()


    args.target = "lm"
    args.prefix_lm_loss = True
    args.batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained("../../TencentPretrain/models/llama_ext/LLaMA-2-7b_v2", device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("../../TencentPretrain/models/llama_ext/LLaMA-2-7b_v2", use_fast=False, trust_remote_code=True)
    args.tokenizer = tokenizer

    t_right, t_wrong, t_no_answer = 0, 0, 0
    right, wrong, no_answer = 0, 0, 0

    import pandas as pd

    with open(args.prediction_path, 'w') as fw:
        for file in os.listdir('../../../falcon/ceval/val'):
            fw.write(file + '\t')
            questions = []
            dev_file = "_".join(file.split('_')[:-1]) + '_dev.csv'

            df = pd.read_csv('../../../falcon/ceval/dev/'+dev_file)
            prefix_list = []
            for index, row in df.iterrows():

                prompt = row['question']

                prompt = prompt + '\n选项：\n'

                prefix = prompt + "A." + row['A'] + '\n' + "B." + row['B'] + '\n' + "C." + row['C'] + \
                         '\n' + "D." + row['D'] + '\n' + '答案： '+ row['answer'] + '\n\n'

                prefix_list.append(prefix)


            df = pd.read_csv('../../../falcon/ceval/val/'+file)
            for index, row in df.iterrows():

                prompt = row['question']
                answer = row['answer']
                answer_texts = ["A." + row['A'], "B." + row['B'], "C." + row['C'], "D." + row['D']]

                prompt = prompt + '\n选项：\n'

                prompt = prompt + "A." + row['A'] + '\n' + "B." + row['B'] + '\n' + "C." + row['C'] + \
                         '\n' + "D." + row['D'] + '\n' + '答案： '

                questions.append((prompt, answer, answer_texts))

            t_right += right
            t_wrong += wrong
            t_no_answer += no_answer

            right, wrong, no_answer = 0, 0, 0
            for que, answer, answer_texts in questions:
                instruction = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize("### Instruction:"))
                response = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize("### Response:"))
                #src = instruction + args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(que)) + response

                prefix1 = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(''.join(random.sample(prefix_list, args.shots))))

                src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(que))
                if args.shots > 0:
                    src = prefix1 + src
                if args.sft:
                    src = instruction + src + response
                seg = [1] * len(src)
                beginning_length = len(src)

                src_tensor, seg_tensor = torch.LongTensor([src]).to(device), torch.LongTensor([seg]).to(device)

                output = model(src_tensor)
                print(output[0][-1].size())

                next_token_logits = F.softmax(output[0][-1])

                print(next_token_logits.size())

                a_prob = next_token_logits[319]
                b_prob = next_token_logits[350]
                c_prob = next_token_logits[315]
                d_prob = next_token_logits[360]

                pred = [a_prob, b_prob, c_prob, d_prob]

                print(pred)

                min_p = 0
                choice = -1
                for i, p in enumerate(pred):
                    if p > min_p:
                        min_p = p
                        choice = i

                char2id = {0:'A', 1:'B', 2:'C', 3:'D'}
                if char2id[choice] == answer:
                    right += 1
                else:
                    wrong += 1

                print(answer, right, wrong)
                print('******************')
                #print(que + "\n")

            fw.write(str(right)+'\t'+str(wrong)+'\t' +str(no_answer)+'\n')
            fw.flush()
        fw.write("total: " + str(t_right)+'\t'+str(t_wrong)+'\t' +str(t_no_answer)+'\n')
        fw.write("acc: " + str(t_right/(t_right+t_wrong)) +'\n')

