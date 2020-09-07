from .detr import DETR


def build_detr_resnet50(num_classes=91, num_queries=100, mask_value=-1., return_decode_sequence=False):
    from .backbone import ResNet50Backbone
    # from chambers.models.resnet import ResNet50Backbone
    return DETR(num_classes=num_classes,
                num_queries=num_queries,
                backbone=ResNet50Backbone(name='backbone/0/body'),
                # backbone=ResNet50Backbone(input_shape=(None, None, 3)),
                mask_value=mask_value,
                return_decode_sequence=return_decode_sequence
                )


def build_detr_resnet50_dc5(num_classes=91, num_queries=100, mask_value=-1., return_decode_sequence=False):
    from .backbone import ResNet50Backbone
    return DETR(num_classes=num_classes,
                num_queries=num_queries,
                backbone=ResNet50Backbone(
                    replace_stride_with_dilation=[False, False, True],
                    name='backbone/0/body'),
                mask_value=mask_value,
                return_decode_sequence=return_decode_sequence
                )


def build_detr_resnet101(num_classes=91, num_queries=100, mask_value=-1., return_decode_sequence=False):
    from .backbone import ResNet101Backbone
    return DETR(num_classes=num_classes,
                num_queries=num_queries,
                backbone=ResNet101Backbone(name='backbone/0/body'),
                mask_value=mask_value,
                return_decode_sequence=return_decode_sequence
                )


def build_detr_resnet101_dc5(num_classes=91, num_queries=100, mask_value=-1., return_decode_sequence=False):
    from .backbone import ResNet101Backbone
    return DETR(num_classes=num_classes,
                num_queries=num_queries,
                backbone=ResNet101Backbone(
                    replace_stride_with_dilation=[False, False, True],
                    name='backbone/0/body'),
                mask_value=mask_value,
                return_decode_sequence=return_decode_sequence
                )
