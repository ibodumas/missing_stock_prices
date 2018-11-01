import connexion
import logging


def _build_api():
    logging.basicConfig(level=logging.INFO)
    app = connexion.App(__name__)
    app.add_api(
        "api_spec.yml",
        strict_validation=True,
        validate_responses=True,
        auth_all_paths=True,
        swagger_ui=True,
    )
    app.run(port=8080)
    return app


try:
    API = _build_api()
except Exception as e:
    logging.critical("Failed to Load App: {}".format(e))
    raise
