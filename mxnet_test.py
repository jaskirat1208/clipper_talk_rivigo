import requests, json, numpy as np
def get_test_point():
    return [np.random.randint(255) for _ in range(785)]
# time.sleep(5)

app_name = "mxnet-test1"
model_name = "mxnet-model1"


print( get_test_point())
url = "http://127.0.0.1:1337/"+ app_name + "/predict"
print(url)
headers = {"Content-type": "application/json"}
out = requests.post(url, headers=headers, data=json.dumps({"input": get_test_point()})).json()
print(out)
