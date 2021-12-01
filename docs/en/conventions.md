# Conventions

Please check the following conventions if you would like to modify MMFlow as your own project.

## Optical flow visualization

In MMFlow, we render the optical flow following this color wheel from [Middlebury flow dataset](https://vision.middlebury.edu/flow/). Smaller vectors are lighter and color represents the direction.

<div align=center>
<img src="https://raw.githubusercontent.com/open-mmlab/mmflow/e9ffff6a01dc8a4770871e5ece05637c7893da8a/resources/color_wheel.png">
</div>

## Return Values

In MMFlow, a `dict` containing losses will be returned by `model(**data, test_mode=False)`, and
a `list` containing a batch of inference results will be returned by `model(**data, test_mode=True)`.
As some methods will predict flow with different direction or occlusion mask, the item type of inference results
is `Dict[str=ndarray]`.

For example in `PWCNetDecoder`,

```python
@DECODERS.register_module()
class PWCNetDecoder(BaseDecoder):

    def forward_test(
        self,
        feat1: Dict[str, Tensor],
        feat2: Dict[str, Tensor],
        H: int,
        W: int,
        img_metas: Optional[Sequence[dict]] = None
    ) -> Sequence[Dict[str, ndarray]]:
        """Forward function when model testing.

        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image.
            H (int): The height of images after data augmentation.
            W (int): The width of images after data augmentation.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.
        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """

        flow_pred = self.forward(feat1, feat2)
        flow_result = flow_pred[self.end_level]

        # resize flow to the size of images after augmentation.
        flow_result = F.interpolate(
            flow_result, size=(H, W), mode='bilinear', align_corners=False)
        # reshape [2, H, W] to [H, W, 2]
        flow_result = flow_result.permute(0, 2, 3,
                                          1).cpu().data.numpy() * self.flow_div

        # unravel batch dim,
        flow_result = list(flow_result)
        flow_result = [dict(flow=f) for f in flow_result]

        return self.get_flow(flow_result, img_metas=img_metas)

    def forward_train(self,
                      feat1: Dict[str, Tensor],
                      feat2: Dict[str, Tensor],
                      flow_gt: Tensor,
                      valid: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Forward function when model training.

        Args:
            feat1 (Dict[str, Tensor]): The feature pyramid from the first
                image.
            feat2 (Dict[str, Tensor]): The feature pyramid from the second
                image.
            flow_gt (Tensor): The ground truth of optical flow from image1 to
                image2.
            valid (Tensor, optional): The valid mask of optical flow ground
                truth. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """

        flow_pred = self.forward(feat1, feat2)
        return self.losses(flow_pred, flow_gt, valid=valid)

    def losses(self,
               flow_pred: Dict[str, Tensor],
               flow_gt: Tensor,
               valid: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute optical flow loss.

        Args:
            flow_pred (Dict[str, Tensor]): multi-level predicted optical flow.
            flow_gt (Tensor): The ground truth of optical flow.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        loss = dict()
        loss['loss_flow'] = self.flow_loss(flow_pred, flow_gt, valid)
        return loss

```
