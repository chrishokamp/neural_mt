import yaml
import os


# define custom tag handler to join filepaths
def path_join(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.join(*[str(i) for i in seq])


# custom tag handler to format strings
def format_str(loader, node):
    seq = loader.construct_sequence(node)
    base_str = seq[0]
    format_vars = seq[1:]
    return base_str.format(*format_vars)

# register the tag handlers
yaml.add_constructor('!path_join', path_join)
yaml.add_constructor('!format_str', format_str)


def get_config(config_file):
    return yaml.load(config_file)
