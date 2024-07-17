import torch
from TTS.api import TTS
import requests
import json
import os
parent_directory_path=os.path.dirname(os.path.abspath(__file__))
print(parent_directory_path)
class t2s:
    
    def __init__(self):
        ##LLM API
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parent_directory_path=parent_directory_path
        self.tts = TTS(
            model_path=self.parent_directory_path+"/model/xtts/",
            config_path=self.parent_directory_path+"/model/xtts/config.json",
            progress_bar=False
                ).to(self.device)

    def get_access_token(self):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        API_KEY = "UvlFgRSCcegpFwK46MhcTW33"
        SECRET_KEY = "YJLw7zgOyVA6qneo8CY9pVl9TjxDpMPw"
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))

    def  get_answer(self,message):
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k?access_token=" + self.get_access_token()
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ],
            "temperature": 0.95,
            "top_p": 0.8,
            "penalty_score": 1.5,
            "disable_search": False,
            "enable_citation": False,
            "response_format": "text"
        })
        headers = {
            'Content-Type': 'application/json'
        }
        print("开始获取")
        response = requests.request("POST", url, headers=headers, data=payload)
        res=json.loads(response.text)
        return  res
    def mytts(self):

            # Example loading a model from a path:
            #     >>> tts = TTS(model_path="/path/to/checkpoint_100000.pth", config_path="/path/to/config.json", progress_bar=False, gpu=False)
            #     >>> tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path="output.wav")
        text=input("请输入：")
        response=self.get_answer(text)
        print(response["result"])
        self.tts.tts_to_file(text=response["result"], speaker_wav=self.parent_directory_path+"/model/xtts/samples_zh-cn-sample.wav",language= 'zh-cn',file_path="../LiveSpeechPortraits/output.wav")

if __name__=='__main__':
    myt2s=t2s()
    myt2s.mytts()
