
# encoding:utf-8

import requests
import sys
import json
import base64

'''
网络图片文字识别
'''
# 保证兼容python2以及python3
IS_PY3 = sys.version_info.major == 3
if IS_PY3:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.parse import quote_plus

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

API_KEY = '9GsUi84NnbYhcelpZKw2sYp0'

SECRET_KEY = 'vQ1211ty85Ip13KdhctpHK8qcnnDrsky'




"""  TOKEN start """
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'


"""
    获取token
"""
def fetch_token():
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    if (IS_PY3):
        post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        print(err)
    if (IS_PY3):
        result_str = result_str.decode()


    result = json.loads(result_str)

    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not 'brain_all_scope' in result['scope'].split(' '):
            print ('please ensure has check the  ability')
            exit()
        return result['access_token']
    else:
        print ('please overwrite the correct API_KEY and SECRET_KEY')
        exit()

"""
网络图片识别标准版: https://aip.baidubce.com/rest/2.0/ocr/v1/webimage
网络图片识别含位置版：  https://aip.baidubce.com/rest/2.0/ocr/v1/webimage_loc
"""
request_url = " https://aip.baidubce.com/rest/2.0/ocr/v1/webimage_loc"
# 二进制方式打开图片文件
f = open('../test_data/020463.jpg', 'rb')
img = base64.b64encode(f.read())

params = {"image":img}
access_token = fetch_token()
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)
if response:
    print (response.json())