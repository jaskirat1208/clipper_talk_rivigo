import logging, xgboost as xgb, numpy as np
from clipper_admin import ClipperConnection, DockerContainerManager
cl = ClipperConnection(DockerContainerManager())
cl.connect()


app_name = 'xgboost-test2'
model_name = 'xgboost-model2'
# We will register it to deploy an xgboost model.
cl.register_application(app_name, 'integers', 'default_pred', 100000)


def get_test_point():
    return [np.random.randint(255) for _ in range(784)]


# Create a training matrix.
dtrain = xgb.DMatrix(get_test_point(), label=[0])
# We then create parameters, watchlist, and specify the number of rounds
# This is code that we use to build our XGBoost Model, and your code may differ.
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
watchlist = [(dtrain, 'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)




def predict(xs):
    return bst.predict(xgb.DMatrix(xs))



from clipper_admin.deployers import python as python_deployer
# We specify which packages to install in the pkgs_to_install arg.
# For example, if we wanted to install xgboost and psycopg2, we would use
# pkgs_to_install = ['xgboost', 'psycopg2']
python_deployer.deploy_python_closure(cl, name=model_name, version=1, input_type="integers", func=predict, pkgs_to_install=['xgboost'])


cl.link_model_to_app(app_name, model_name)

