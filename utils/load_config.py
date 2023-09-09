import yaml

with open("utils/constants.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)