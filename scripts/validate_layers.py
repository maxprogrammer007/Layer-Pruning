# scripts/validate_layers.py

def validate_layer_names(model, config):
    """
    Check if each layer name in config["layers_to_hook"] exists in the model.
    Returns two lists: valid_layers, invalid_layers
    """
    valid = []
    invalid = []

    # For MobileNet, we target model.backbone.features
    model_layers = dict(model.backbone.features.named_children())

    for lname in config["layers_to_hook"]:
        short_name = lname.split(".")[-1]  # e.g., features.14 → 14
        if short_name in model_layers:
            valid.append(lname)
        else:
            invalid.append(lname)

    return valid, invalid
