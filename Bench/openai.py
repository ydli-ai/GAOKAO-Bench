import requests
import time
import openai
from random import choice
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class  OpenaiAPI:
    def __init__(self, api_key_list, model_name="gpt-3.5-turbo", temperature=0.3, max_tokens=1024):
        self.api_key_list = api_key_list
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.model = AutoModelForCausalLM.from_pretrained("../../TencentPretrain/models/llama_ext/LLaMA-2-7b_v2", device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("../../TencentPretrain/models/llama_ext/LLaMA-2-7b_v2", use_fast=False, trust_remote_code=True)


    def forward(self, prompt, question)->str:
        """
        """
        prompt = f"### Instruction:{prompt + question}  ### Response:"
        if len(prompt) > 512:
            prompt = prompt[-512:]
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        generate_ids = self.model.generate(inputs.input_ids, do_sample=True, max_new_tokens=128, top_k=10, top_p=0.85, temperature=1, repetition_penalty=1.15, eos_token_id=2, bos_token_id=1, pad_token_id=0)
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.lstrip(prompt)

        return response
    
    def postprocess(self, output):
        """
        """
        model_output = None

        if self.model_name == "gpt-3.5-turbo":
            model_output = output['choices'][0]['message']['content']

        elif self.model_name == 'text-davinci-003':
            model_output = output['choices'][0]['text']

        if not model_output:
            print("Warning: Empty Output ") 
        return model_output

    def __call__(self, prompt:str, question:str):
        return self.forward(prompt, question)


def test(model, prompt:str, question:str):


    response = model(prompt, question)

    return response


if __name__ == "__main__":
    api_key_list = ["openai_api_key"]
    model_api = OpenaiAPI(api_key_list, model_name="gpt-3.5-turbo")
    data_example = {
            "year": "2010",
            "category": "（新课标Ⅰ）",
            "question": "21. --- Have  you finished  reading  Jane  Eyre ? \n--- No, I        my homework  all day yesterday . \nA. was doing  B. would  do C. had done  D. do\n",
            "answer": [
                "A"
            ],
            "analysis": "【解答】 答案 A． was/were  doing，表示过去的某个时间点或时间段正在做某事\n，根据句意，我没有读完简爱，我昨天一天一直在写家庭作业． 故选 A． \n【点评】\n",
            "index": 0,
            "score": 1
        }
    choice_question = data_example['question']
    choice_prompt = "请你做一道英语选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下："

    result = test(model_api, choice_prompt, choice_question)

    print("Model output:\n" + result)
    
