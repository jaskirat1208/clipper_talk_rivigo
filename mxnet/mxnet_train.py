import time
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.mxnet import deploy_mxnet_model
import mxnet as mx

app_name = "mxnet"
model_name = "mxnet"
version = 1

def predict(model, xs):
    data_iter = mx.io.NDArrayIter(xs)
    preds = model.predict(data_iter)
    preds = [preds[0]]
    return [str(p) for p in preds]

clipper_conn = ClipperConnection(DockerContainerManager())
clipper_conn.connect()
# clipper_conn.start_clipper()


# To start clipper, use clipper_conn.start_clipper function
time.sleep(5)

clipper_conn.register_application(app_name, "integers",
                                              "default_pred", 100000)
time.sleep(1)

train_path = "../trainingdata.csv"
data_iter = mx.io.CSVIter(
            data_csv=train_path, data_shape=(785, ), batch_size=1)

# Create a MXNet model
# Configure a two layer neuralnetwork
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type='relu')
fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(fc2, name='softmax')

# Initialize the module and fit it
mxnet_model = mx.mod.Module(softmax)
mxnet_model.fit(data_iter, num_epoch=0)

train_data_shape = data_iter.provide_data

deploy_mxnet_model(
        clipper_conn,
        model_name,
        version,
        "integers",
        predict,
        mxnet_model,
        train_data_shape,
        batch_size=1)
time.sleep(5)

clipper_conn.link_model_to_app(app_name, model_name)
time.sleep(5)

