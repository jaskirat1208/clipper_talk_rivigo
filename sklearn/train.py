from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.python import deploy_python_closure
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

app_name = "sklearn-test16"
model_name = "sklearn-model16"

clipper_conn = ClipperConnection(DockerContainerManager())

# Connect to an already-running Clipper cluster
clipper_conn.connect()

def center(xs):
    means = np.mean(xs, axis=0)
    return xs - means

# centered_xs = center(xs)
# model = sklearn.linear_model.LogisticRegression()
# ys = np.array()
# model.fit(centered_xs, ys)
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)

# Note that this function accesses the trained model via closure capture,
# rather than having the model passed in as an explicit argument.
def centered_predict(inputs):
    # preds = reg.predict(center(inputs))
    preds = clf.predict(inputs)
    return [str(p) for p in preds]


print("REGISTERING APPLICATION")
clipper_conn.register_application(name=app_name, input_type="doubles", default_output="-1.0", slo_micros=10000000)


print("REGISTERING DONE. NOW DEPLOYING")
deploy_python_closure(
    clipper_conn,
    name=model_name,
    version=1,
    input_type="doubles",
    func=centered_predict)

print("DEPLOYING SUCCESS. NOW LINKING MODEL TO APPLICATION")

clipper_conn.link_model_to_app(app_name=app_name, model_name=model_name)
