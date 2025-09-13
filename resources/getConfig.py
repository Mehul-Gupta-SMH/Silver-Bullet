import yaml


def __get_config__(config_path='./resources/config.yaml'):
    """ Load configuration from a YAML file.
        Args:
            config_path (str): Path to the YAML configuration file.
        Returns: dict: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except:
        with open(r'C:\Users\mehul\Documents\Projects - GIT\SilverBullet\Silver-Bullet\resources\config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    return config


def getVal(env='DEVELOPMENT'):
    """ Fetch configuration values based on the specified environment.
        Args:
            env (str): The environment for which to fetch the configuration ('dev', 'prod', etc.).
        Returns: dict: Configuration dictionary for the specified environment.
    """
    config = __get_config__()
    if env in config:
        return config[env]


if __name__ == "__main__":
    config = getVal()
    print(config)