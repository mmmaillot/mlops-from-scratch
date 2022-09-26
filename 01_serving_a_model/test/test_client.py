from fastapi.testclient import TestClient
from main import app


def test_is_response_has_correct_form():
    client = TestClient(app)
    response = client.post("/label",
                           json={"sepal_length": 0,
                                 "sepal_width": 0,
                                 "petal_length": 0,
                                 "petal_width": 0}).json()
    assert "label" in response
    assert response["label"] in [0, 1, 2]
