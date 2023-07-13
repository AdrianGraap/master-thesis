import requests
import json
from elasticsearch import Elasticsearch
import sys

# es = Elasticsearch(hosts=['http://localhost:9200'])
# docType = "doc"
# indexName = 'wikidataentityindex'
# query = 'wife'
#
if __name__ == "__main__":
    headers = {'Content-type': 'application/json'}
    q = 'Who is the president of germany?'
    data = {"nlquery": q}
    data = json.dumps(data, ensure_ascii=True)
    data = data.encode('utf-8')
    # data = f'{{\"nlquery\":\"{q}\"}}'
    r = requests.post('http://localhost:4999/processQuery', data=data, headers=headers)
    print(r.text)

# Who developed games based on the Cars series?
# Give me the count of artist in the company whose Artist is Jean- François Coté ?