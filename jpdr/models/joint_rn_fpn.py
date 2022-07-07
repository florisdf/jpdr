from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import TwoMLPHead

from .joint_rcnn import JointRCNN


class JointResNetFPN(JointRCNN):
    def __init__(
        self,
        *args,
        backbone_name,
        pretrained,
        trainable_layers=3,
        roi_output_size=14,
        box_head_out_channels=1024,
        featmap_names=['0', '1', '2', '3'],
        anchor_sizes=((16,), (32,), (64,), (128,), (256,)),
        **kwargs
    ):
        """
        Args:
            backbone_name (string): resnet architecture. Possible values are
                'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                'wide_resnet50_2', 'wide_resnet101_2'
            pretrained (bool): If True, returns a model with backbone
                pre-trained on Imagenet
            trainable_layers (int): number of trainable (not frozen) resnet
                layers starting from final block.  Valid values are between 0
                and 5, with 5 meaning all backbone layers are trainable.
            roi_output_size (int): Output size of the RoI pooling layer.
            box_head_out_channels (int): The number of output channels of the
                box head.
            featmap_names: The names of the feature maps (in the ordered dict
                of feature maps returned by the backbone) that will be used for
                pooling.
            anchor_sizes: The anchor sizes to use for each feature pyramid
                level, e.g. `((128,), (256,), (512,))`.
            args: See `JointRCNN`
            kwargs: See `JointRCNN`
        """
        backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            trainable_layers=trainable_layers
        )

        out_channels = backbone.out_channels
        box_head = TwoMLPHead(
            out_channels * roi_output_size ** 2,
            box_head_out_channels
        )

        super().__init__(
            *args,
            backbone=backbone,
            box_head=box_head,
            box_head_out_channels=box_head_out_channels,
            featmap_names=featmap_names,
            anchor_sizes=anchor_sizes,
            roi_output_size=roi_output_size,
            **kwargs
        )
