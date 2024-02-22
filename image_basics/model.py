"""
Image model retrieval and initialization.
"""
import timm

_DEFAULT_MODEL = "convnext"


def _get_model_name(name: str = _DEFAULT_MODEL):
    """Retrieve official model name based on simplified tag."""
    if name == "convnext":
        return "convnext_tiny.in12k_ft_in1k"
    if name == "regnety":
        return "regnety_002.pycls_in1k"
    else:
        raise ValueError(f"{name} is not a valid model str")


def _get_model_config(name: str = _DEFAULT_MODEL):
    """Retrieve timm model pretrained model config dict."""
    return timm.get_pretrained_cfg(_get_model_name(name)).to_dict()


def create_model(name: str = _DEFAULT_MODEL, pretrained=True, num_classes=2):
    timm_name = _get_model_name(name)
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
