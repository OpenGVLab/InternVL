# How to use InternVL API?

## 1. Official API of InternVL2

We encourage everyone to use our API for research. For better management, please submit ([English application form](https://docs.google.com/forms/d/e/1FAIpQLSfMCzhPr1OOEKau_6jwTU0EiZMSFckDo-HMlc_hUudhF_97rw/viewform?usp=sf_link))/([中文申请表](https://wj.qq.com/s2/14910502/25a4/)) to obtain free API access.

## 2. Community-Host API of InternVL 1.5

https://rapidapi.com/adushar1320/api/internvl-chat

## 3. Examples

```python
import requests

url = "" # （API）
api_key = ""  # （KEY）


# example
file_paths = [
    "./image/1f0537f2.png",
    "./image/3c0c0aaa.png",
    "./6b374376.png"
]
question = "Describe the three images in detail." # (Question)


'''
# text example
file_paths = []
question = "describe beijing"
'''

files = [('files', open(file_path, 'rb')) for file_path in file_paths]
data = {
    'question': question,
    'api_key': api_key
}

try:
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        print("Response:", response.json().get("response", "No response key found in the JSON."))
    else:
        print("Error:", response.status_code, response.text)
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
```


