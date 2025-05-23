#Taken from ETH Repo and modified
import yaml
from on_track_sysid.dotdict import DotDict

def get_dict(model_name):
    model, tire = model_name.split("_")
    package_path = 'src/on_track_sysid'
    with open(f'{package_path}/models/{model}/{model_name}.txt', 'rb') as f:
        params = yaml.load(f, Loader=yaml.Loader)
    return params

def get_dotdict(model_name):
    dict = get_dict(model_name)
    params = DotDict(dict)
    return params