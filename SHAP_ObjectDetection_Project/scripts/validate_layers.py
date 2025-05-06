# scripts/validate_layers.py

def validate_layer_names(model, config):
    backbone = model.backbone
    valid = []
    invalid = []

    for lname in config["layers_to_hook"]:
        try:
            parts = lname.split(".")
            mod = backbone
            for p in parts:
                if p.isdigit():
                    mod = mod[int(p)]
                else:
                    mod = getattr(mod, p)
            valid.append(lname)
        except Exception:
            invalid.append(lname)

    return valid, invalid
