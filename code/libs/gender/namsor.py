import requests
import json

class Namsor(object):

    URI_NAME = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderChineseNamePinyinBatch"

    HEADERS = {
                "X-API-KEY": "your-api-key",
                "Accept": "application/json",
                "Content-Type": "application/json"
              }

    def __init__(self, api_key):
        self.api_key = api_key


    def get_header(self):
        h = self.HEADERS
        h["X-API-KEY"] = self.api_key
        return h
    
    def infer_gender_from_chinese_given_name(self, given_name):
        payload = {
            "personalNames": [
                {
                "firstName": given_name,
                }
            ]
        }
        response = requests.post(self.URI_NAME, headers=self.get_header(), json=payload)
        return response.text