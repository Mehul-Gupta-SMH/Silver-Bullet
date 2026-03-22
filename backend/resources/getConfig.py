import os
import yaml
from pathlib import Path


def __get_config__(config_path=None):
    """Load configuration from a YAML file.

    Searches for config.yaml relative to this file's location so the project
    can be run from any working directory.

    Args:
        config_path (str | None): Override path to the YAML config file.
    Returns:
        dict: Configuration dictionary.
    Raises:
        FileNotFoundError: If config.yaml cannot be found.
    """
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def getVal(env='DEVELOPMENT'):
    """Fetch configuration values for the given environment.

    Environment variables take precedence over the YAML file.
    Recognised env vars (all optional):
        SB_HF_TOKEN       — HuggingFace API token
        SB_OPENAI_TOKEN   — OpenAI API token
        SB_API_KEY        — Internal API key
        SB_API_URL        — Service URL

    Args:
        env (str): Environment name matching a top-level key in config.yaml
                   ('DEVELOPMENT', 'STAGING', 'PRODUCTION').
    Returns:
        dict: Configuration dictionary for the requested environment.
    """
    config = __get_config__()
    values = config.get(env, {})

    # Environment variable overrides — never commit real secrets to config.yaml
    env_overrides = {
        'hf_token':     os.environ.get('SB_HF_TOKEN'),
        'openai_token': os.environ.get('SB_OPENAI_TOKEN'),
        'api_key':      os.environ.get('SB_API_KEY'),
        'url':          os.environ.get('SB_API_URL'),
    }
    for key, val in env_overrides.items():
        if val is not None:
            values[key] = val

    return values


if __name__ == "__main__":
    config = getVal()
    # Print without exposing token values
    safe = {k: ('***' if 'token' in k or 'key' in k else v) for k, v in config.items()}
    print(safe)
