# Load model directly
from typing import List
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from resources.getConfig import getVal
from openai import OpenAI
login(token=getVal()['hf_token'])


class EntityResolver:
    def __init__(self, model: str = "google/gemma-3-270m"):
        self.tokenizer = None
        self.model = None
        self.MODEL = model
        self.__model_selection__()
        self.__config__ = getVal()
        self.__load_model__()

    def __model_selection__(self):
        if self.MODEL in ['gpt-4o', 'gpt-4o-mini', 'gpt-4o-mini-2024-08-06', 'gpt-4o-2024-08-06']:
            self.MODEL_TYPE = 'api'
        else:
            self.MODEL_TYPE = 'local'

    def __load_model__(self):
        if self.MODEL_TYPE == 'api':
            self.model = OpenAI(api_key=self.__config__['openai_token'])
        elif self.MODEL_TYPE == 'local':
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL,
                use_fast=True,
                add_eos_token=True,
                add_bos_token=True,
                token=self.__config__['hf_token'],
                cache_dir=f'/Preprocess/coref/model/generative/{self.MODEL}'
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                cache_dir=f'/Preprocess/coref/model/generative/{self.MODEL}'
            )
        else:
            raise NotImplementedError("Model not supported")

    @staticmethod
    def __get_prompt_local__(text: str) -> List[dict[str, list[dict[str, str]]]]:
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": """You are an expert in natural language processing and coreference resolution. Your task is to replace pronouns in the given text with the correct named entities to enhance clarity and understanding.Ensure that the revised text maintains the original meaning while improving readability."""
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                ]
            }
        ]

    @staticmethod
    def __get_prompt_api__(text: str) -> List[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": """You are an expert in natural language processing and coreference resolution. Your task is to replace pronouns in the given text with the correct named entities to enhance clarity and understanding.Ensure that the revised text maintains the original meaning while improving readability."""
            },
            {
                "role": "user",
                "content": text
            }
        ]

    def __resolve_local__(self, text: str) -> str:
        base_prompt = self.__get_prompt_local__(text)
        inputs = self.tokenizer(base_prompt[0], return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def __resolve_api__(self, text: str) -> str:
        base_prompt = self.__get_prompt_api__(text)
        response = self.model.chat.completions.create(
            model=self.MODEL,
            messages=base_prompt,
            temperature=0.1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=1.2,
        )
        response = json.loads(response.model_dump_json())
        return response['choices'][0]['message']['content']

    def resolve(self, text: str) -> str:
        if self.MODEL_TYPE == 'local':
            return self.__resolve_local__(text)
        elif self.MODEL_TYPE == 'api':
            return self.__resolve_api__(text)
        else:
            raise NotImplementedError("Model not supported")


if __name__ == "__main__":
    resolver = EntityResolver(model="gpt-4o-mini")
    text = "My Name is Mehul. Mehul is a good person. But he can be bad sometimes."
    resolved_text = resolver.resolve(text)
    print(resolved_text)
