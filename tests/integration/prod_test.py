import requests
import json

test_sample = json.dumps({'data': [[19,1,27.9,0,1,3,16884.924]]})
test_sample = str(test_sample)

def test_ml_service(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, test_sample, headers=headers)
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0