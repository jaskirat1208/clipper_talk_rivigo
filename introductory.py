from clipper_admin import ClipperConnection, DockerContainerManager

clipper_conn = ClipperConnection(DockerContainerManager())
# clipper_conn.start_clipper()
clipper_conn.connect()

app_name = "first"
model_name = "first"

clipper_conn.register_application(name=app_name, input_type="doubles", default_output="-1.0", slo_micros=100000)
clipper_conn.get_all_apps()



def feature_sum(xs):
	return [str((sum(x))) for x in xs ]




from clipper_admin.deployers import python as python_deployer


python_deployer.deploy_python_closure(clipper_conn, name=model_name, version=1, input_type="doubles", func=feature_sum)
clipper_conn.link_model_to_app(app_name=app_name, model_name=model_name)