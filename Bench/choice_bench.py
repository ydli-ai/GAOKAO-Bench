from bench_function import choice_test, get_api_key, export_distribute_json, export_union_json
import os
import json
import time

if __name__ == "__main__":
    with open("MCQ_prompt.json", "r") as f:
        data = json.load(f)
    f.close()

    
for i in range(len(data['examples'])):
    directory = "../data"

    api_key_filename = "your_api_key.txt"
    api_key_list = get_api_key(api_key_filename, start_num=0, end_num = 1)
    

    model_name = 'gpt-3.5-turbo'
    temperature = 0.3
    keyword = data['examples'][i]['keyword']

    question_type = data['examples'][i]['type']
    zero_shot_prompt_text = data['examples'][i]['prefix_prompt']
    print(keyword)
    print(question_type)
    #TODO:
    model_api=None

    export_distribute_json(
        model_api, 
        api_key_list, 
        model_name, 
        temperature, 
        directory, 
        keyword, 
        zero_shot_prompt_text, 
        question_type, 
        parallel_num=5
    )

    export_union_json(
        directory, 
        model_name, 
        keyword,
        zero_shot_prompt_text,
        question_type
    )
    


