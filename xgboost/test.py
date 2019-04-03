import requests, json
import logging, xgboost as xgb, numpy as np
from clipper_admin import ClipperConnection, DockerContainerManager

# Get Address
cl = ClipperConnection(DockerContainerManager())
cl.connect()
addr = cl.get_query_addr()
# Post Query


def get_test_point():
    return [np.random.randint(255) for _ in range(784)]

response = requests.post(
    "http://%s/%s/predict" % (addr, 'xgboost-test'),
    headers={"Content-type": "application/json"},
    data=json.dumps({
        'input': get_test_point()
    }))
result = response.json()
if response.status_code == requests.codes.ok and result["default"]:
    print('A default prediction was returned.')
elif(response.status_code != requests.codes.ok):
    print(result)
    raise BenchmarkException(response.text)
else:
    print('Prediction Returned:', result)