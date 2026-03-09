import json
import numpy as np
from utility.transforms import T

def jsonify_dict(d):
    jsonified_dict = {}

    for key in d:
        # Convert all matrices to nested lists for JSON serialization
        if isinstance(d[key], np.ndarray):
            jsonified_dict[key] = d[key].tolist()
        elif isinstance(d[key], T):
            jsonified_dict[key] = d[key].jsonify()
        elif isinstance(d[key], list) and len(d[key]) > 0:
            if isinstance(d[key][0], np.ndarray):
                jsonified_dict[key] = [item.tolist() for item in d[key]]
            elif isinstance(d[key][0], T):
                jsonified_dict[key] = [item.jsonify() for item in d[key]]
            else:
                # Nested list
                jsonified_dict[key] = d[key]

        elif isinstance(d[key], dict):
            jsonified_dict[key] = jsonify_dict(d[key])
        else:
            jsonified_dict[key] = d[key]

    return jsonified_dict

def write_to_json(data, file_path):    
    with open(file_path, 'w') as f:
        json.dump(jsonify_dict(data), f, indent=2)