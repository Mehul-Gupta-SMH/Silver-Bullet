from Postprocess.__addpad import pad_matrix


def apply(feature_map, **kwargs):
    """ Apply padding to the feature map matrix. """
    feature_map = pad_matrix(feature_map,
                             target_size=kwargs.get('target_size', 64),
                             pad_value=kwargs.get('pad_value', 0)
                             )
    return feature_map