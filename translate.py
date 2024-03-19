import requests
from dotenv import load_dotenv
import os

class Translate:
    

    def __init__(self) -> None:
        load_dotenv()
        self.__key = os.getenv("KEY")
        self.source = None

    # def detect_language(self, text) -> requests.Response:
    #     link = (
    #         f"https://translation.googleapis.com/language/translate/v2/detect?"
    #         f"q={text}&"
    #         f"key={self.__key}"
    #     )

    #     request = requests.post(link)
    #     return request


    def translate_text(self, text) -> requests.Response.json:

        if not self.source:
            target = 'en'
        else:
            target = self.source

        link = (
                f"https://translation.googleapis.com/language/translate/v2?"
                f"q={text}&"
                f"target={target}&"
                f"key={self.__key}"
        )
        
        request = requests.post(link).json()
        self.source = request["data"]["translations"][0]["detectedSourceLanguage"]
        return request


# text = "dzien dobry, jak panu mija dzionek?"

# t = Translate()
# translate = t.translate_text(text)
# print(translate)
# translate = translate["data"]["translations"][0]["translatedText"]
# print(translate)
# t2 = t.translate_text(translate)
# print(t2)
# t2 = t2["data"]["translations"][0]["translatedText"]
# print(t2)



