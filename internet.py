import requests
import json


def main():
   # resp = requests.get('https://api.tianapi.com/guonei/?key=610ce1a842c8534c3e3b47b16ad6ecd4&num=10')
    resp = requests.get('https://api.tianapi.com/txapi/yuanqu/index?key=610ce1a842c8534c3e3b47b16ad6ecd4&num=10')
    data_model = json.loads(resp.text)
    for news in data_model['newslist']:
        print(news['content'])


if __name__ == '__main__':
    main()