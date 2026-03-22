from backend.Postprocess.__addpad import resize_matrix, TARGET_SIZE


def apply(feature_map, **kwargs):
    """Resize the feature map matrix to a fixed square via bilinear interpolation."""
    return resize_matrix(feature_map, target_size=kwargs.get('target_size', TARGET_SIZE))
