
# import os
# import yaml

# class ModelManager:
#     """
#     A singleton class for managing API keys.

#     This class provides functionality to allocate random API keys
#     from the available keys read from a configuration file.

#     Methods:
#         allocate_key(): Allocate a random API key from the available keys.
#     """

#     _instance = None

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(ModelManager, cls).__new__(cls)
#             _current_dir = os.path.dirname(os.path.abspath(__file__))
#             _config_path = os.path.join(_current_dir, "../config/llm_config.yaml")
#             try:
#                 with open(_config_path, "r") as config_file:
#                     cls._config = yaml.safe_load(config_file)
#             except FileNotFoundError:
#                 print(f"Error: Configuration file '{_config_path}' not found.")
#         return cls._instance

#     def allocate(self, model_family: str = "GPT", model_name_override: str = None):
#         api_base = self._config["api_base"][model_family]
#         api_key = self._config["api_key"][model_family]
#         model = model_name_override if model_name_override else self._config["model"][model_family]
#         return api_base, api_key, model


# model_manager = ModelManager()

# if __name__ == "__main__":
#     print(model_manager.allocate(model_family="QWEN"))


import os
import yaml


def get_llm_service_tier() -> str:
    """Return the configured LLM service tier, accepting common env aliases."""
    for name in (
        "GSI_LLM_SERVICE_TIER",
        "OPENROUTER_SERVICE_TIER",
        "LLM_SERVICE_TIER",
        "SERVICE_TIER",
        "SERVICETIER",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _get_first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bool(value: str) -> bool | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def get_openrouter_provider() -> dict:
    """Return OpenRouter provider routing preferences from environment variables."""
    provider = {}
    only = _split_csv(
        _get_first_env(
            (
                "GSI_LLM_PROVIDER_ONLY",
                "OPENROUTER_PROVIDER_ONLY",
                "LLM_PROVIDER_ONLY",
            )
        )
    )
    order = _split_csv(
        _get_first_env(
            (
                "GSI_LLM_PROVIDER_ORDER",
                "OPENROUTER_PROVIDER_ORDER",
                "LLM_PROVIDER_ORDER",
            )
        )
    )
    allow_fallbacks = _parse_bool(
        _get_first_env(
            (
                "GSI_LLM_ALLOW_FALLBACKS",
                "OPENROUTER_ALLOW_FALLBACKS",
                "LLM_ALLOW_FALLBACKS",
            )
        )
    )

    if only:
        provider["only"] = only
    if order:
        provider["order"] = order
    if allow_fallbacks is not None:
        provider["allow_fallbacks"] = allow_fallbacks
    return provider


def get_openrouter_extra_body() -> dict:
    """Return extra OpenRouter request fields configured through environment variables."""
    extra_body = {}
    service_tier = get_llm_service_tier()
    provider = get_openrouter_provider()
    if service_tier:
        extra_body["service_tier"] = service_tier
    if provider:
        extra_body["provider"] = provider
    return extra_body


class ModelManager:
    """
    A singleton class for managing API keys.

    This class provides functionality to allocate random API keys
    from the available keys read from a configuration file.

    Methods:
        allocate_key(): Allocate a random API key from the available keys.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            _current_dir = os.path.dirname(os.path.abspath(__file__))
            _config_path = os.path.join(_current_dir, "../config/llm_config.yaml")
            try:
                with open(_config_path, "r") as config_file:
                    cls._config = yaml.safe_load(config_file)
            except FileNotFoundError:
                print(f"Error: Configuration file '{_config_path}' not found.")
        return cls._instance

    def allocate(self, model_family: str = "GPT", model_name_override: str = None):
        api_base = os.environ.get("GSI_LLM_API_BASE") or self._config["api_base"][model_family]
        api_key = os.environ.get("GSI_LLM_API_KEY") or self._config["api_key"][model_family]
        model = (
            model_name_override
            or os.environ.get("GSI_LLM_MODEL")
            or self._config["model"][model_family]
        )
        return api_base, api_key, model


model_manager = ModelManager()

if __name__ == "__main__":
    print(model_manager.allocate(model_family="QWEN"))