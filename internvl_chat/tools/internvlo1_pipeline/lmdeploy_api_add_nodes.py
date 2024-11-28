import os
import re
import requests

url = 'http://10.140.37.1:8000/nodes'

def check_status():
    headers = {'accept': 'application/json'}
    response = requests.get(os.path.join(url, 'status'), headers=headers)
    print(response.text)

def parse_to_list(input_string):
    # 提取主机地址的前三部分和范围部分
    match = re.match(r".*-(\d+)-(\d+)-(\d+)-\[(.*)]", input_string)
    if not match:
        raise ValueError("Input format is incorrect")

    part1, part2, part3, ranges = match.groups()
    base_ip = f"{part1}.{part2}.{part3}"

    # 解析范围部分
    result = []
    for part in ranges.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            result.extend([f"{base_ip}.{i}" for i in range(start, end + 1)])
        else:
            result.append(f"{base_ip}.{int(part)}")

    return result

def add_nodes():
    sub_url_str = 'SH-IDC1-10-140-37-[39-40,42,44,46-48,51,54,58,63,68-73,75,77-80,135,151]'
    sub_url = parse_to_list(sub_url_str)
    print(f'{len(sub_url)=}')

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    sub_url = [
        f"http://{u}:23333"
        for u in sub_url
    ]
    for u in sub_url:
        data = {"url": u}
        response = requests.post(os.path.join(url, 'add'), headers=headers, json=data)
        print(u,response.text)

add_nodes()
check_status()
