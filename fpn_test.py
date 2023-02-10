
import torch
import torchvision

# retinanet_resnet50_fpn_v2 is not available in this torchvision 
model = torchvision.models.detection.retinanet_resnet50_fpn(progress = True, 
                                                            trainable_backbone_layers = 5,
                                                            num_classes = 1)
print(model)
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model['backbone'](x)
for name, param in model.state_dict().items():
    print(name)
# print(predictions)
