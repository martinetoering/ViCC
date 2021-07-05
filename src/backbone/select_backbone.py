from .s3dg import S3D
from .r21d import R2Plus1DNet

def select_backbone(network, first_channel=3):
    param = {'feature_size': 1024}
    if network == 's3d':
        model = S3D(input_channel=first_channel)
    elif network == 's3dg':
        model = S3D(input_channel=first_channel, gating=True)
    elif network == 'r21d':
        param['feature_size'] = 512
        model = R2Plus1DNet()
    return model, param
