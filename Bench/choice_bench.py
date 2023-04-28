import sys
import os
parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)

from bench_function import get_api_key, export_distribute_json, export_union_json
import os
import json
import time

class ChatflowAPI:
    def __init__(self, api_key_list):
        self.api_key_list = api_key_list
        self.api_url = "http://127.0.0.1:8888/chat"
        

    def send_request(self, api_key, request:str, context=None):

        self.headers = {
            "Content-Type": "application/json"
        }
        data = {
                "question": request
        }

        if context:
            data["context"] = context

        response = requests.post(self.api_url, headers=self.headers, json=data)
        return response.json()

    def forward(self, request_text:str):
        """
        """
        

        while True:
            try:
                response = self.send_request("api_key", request_text)
                if 'answer' in response.keys():
                    response = response['answer'][0]
                    break

            except Exception as e:
                print('Exception:', e)
                time.sleep(4)
 
        return response

    def __call__(self, prompt, question):
        return self.forward(request_text=prompt+question)


if __name__ == "__main__":
    # Load the MCQ_prompt.json file
    with open("MCQ_prompt.json", "r") as f:
        data = json.load(f)['examples']
    f.close()

    
for i in range(len(data)):
    directory = "../data/Multiple-choice_Questions"

    # get the api_key_list
    #openai_api_key_file = "your openai api key list"
    #openai_api_key_list = get_api_key(openai_api_key_file, start_num=0, end_num=1)
    # moss_api_key_list = [""]
    
    # get the model_name and instantiate model_api
    model_name = 'chatflow-13b'
    model_api = ChatflowAPI("")
    # model_name = 'moss'
    # model_api = MossAPI(moss_api_key_list)

    keyword = data[i]['keyword']
    question_type = data[i]['type']
    zero_shot_prompt_text = data[i]['prefix_prompt']
    print(keyword)
    print(question_type)

    export_distribute_json(
        model_api, 
        model_name, 
        directory, 
        keyword, 
        zero_shot_prompt_text, 
        question_type, 
        parallel_num=5, 
    )

    export_union_json(
        directory, 
        model_name, 
        keyword,
        zero_shot_prompt_text,
        question_type
    )
    


