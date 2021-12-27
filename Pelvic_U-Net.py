from torch import nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F


sigmoid_activation = lambda x: F.sigmoid(x) 


class sa_layer(nn.Module):

    """
    Shuffle attention module class
    
    https://arxiv.org/abs/2102.00240

    @misc{yang2021sanet,
      title={SA-Net: Shuffle Attention for Deep Convolutional Neural Networks}, 
      author={Qing-Long Zhang Yu-Bin Yang},
      year={2021},
      eprint={2102.00240},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
    """

    def __init__(self, input_features, groups=16):

        """
        class initializer

        Args:
            input_features (int): Number of input features (channels) 
            groups (int): Number of groups used in the SA module
        Returns:
            None
        """

        super().__init__()

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.cweight = Parameter(torch.zeros(1, input_features // (2 * groups), 1, 1, 1))
        self.cbias = Parameter(torch.ones(1, input_features // (2 * groups), 1, 1, 1))
        self.sweight = Parameter(torch.zeros(1, input_features // (2 * groups), 1, 1, 1))
        self.sbias = Parameter(torch.ones(1, input_features // (2 * groups), 1, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(input_features // (2 * groups), input_features // (2 * groups))
        
    @staticmethod
    def channel_shuffle(x, groups):

        """
        channel shuffle
        Args:
            x (tensor): feature maps
            groups (int): number of groups in channel shuffle
        Returns:
            x(tensor)
        """

        b, c, d, h, w = x.shape

        x = x.reshape(b, groups, -1, d, h, w)
        x = x.permute(0, 2, 1, 3, 4, 5)

        # flatten
        x = x.reshape(b, -1, d, h, w)

        return x

    def forward(self, x):

        """
        SA module forward path
        Args:
            x (tensor): feature maps
        Returns:
            out (tensor): SA module output
        """

        #get feature map shape
        b, c, d, h, w = x.shape

        #reshape x 
        x = x.reshape(b * self.groups, -1, d, h, w)

        #split x into x0 and x1 for channel and spatial attention
        x_0, x_1 = x.chunk(2, dim=1)

        #channel attention
        xn = self.avg_pool(x_0) #average pooling
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        #spatial attention
        xs = self.gn(x_1) #group normalization
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, d, h, w)

        #shuffle for cross information flow across channel dimension
        out = self.channel_shuffle(out, 2)

        return out

class InitWeights_He(object):

    """
    Weight initializer
    """

    def __init__(self, neg_slope=1e-2):

        """
        class initializer

        Args:
            neg_slope (float)
        Returns:
            None
        """

        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvolutionalBlock(nn.Module):

    """
    Convolutional block using convolution, non-linear activation and BatchNorm.
    """

    def __init__(self, features_in, features_out,
                  convolutional_params=None, normalization_params=None, activation_params=None):
        
        """
        class initializer
        Args:
            features_in (int): number of input features
            features_out (int): number of output features
            convolutional_params (dict): parameters for convolutional layer
            normalization_params (dict): parameters for BatchNorm layer
            activation_params (dict): parameters for LeakyReLU layer
        Returns:
            None
        """

        super().__init__()

        self.conv = nn.Conv3d(features_in, features_out, **convolutional_params)
        self.batchNorm = nn.BatchNorm3d(features_out, **normalization_params)
        self.lrelu = nn.LeakyReLU(**activation_params)

    def forward(self, x):

        """
        forward method
        Args:
            x (tensor): input feature map
        Returns:
            x (tensor): output feature map
        """

        x = self.conv(x)
        return self.batchNorm(self.lrelu(x))        


class StackedBlock(nn.Module):

    """
    Stacks a pre-defined number of convolutional layers into a single block.
    """

    def __init__(self, features_in, features_out, number_convolutions,
                 convolutional_params=None, normalization_params=None, activation_params=None):
        
        """
        class initializer
        Args:
            features_in (int): number of input features
            features_out (int): number of output features
            number_convolutions (int): number of convolutional blocks in a single stacked block
            convolutional_params (dict): parameters for convolutional layer
            normalization_params (dict): parameters for BatchNorm layer
            activation_params (dict): parameters for LeakyReLU layer
        Returns:
            None
        """

        self.channels_in = features_in
        self.channels_out = features_out

        super().__init__()

        self.blocks = nn.Sequential(
            *([ConvolutionalBlock(features_in, features_out,
                           convolutional_params,
                           normalization_params, activation_params)] +
              [ConvolutionalBlock(features_out, features_out, 
                           convolutional_params, normalization_params,
                            activation_params) for _ in range(number_convolutions - 1)]))

    def forward(self, x):

        """
        forward method
        Args:
            x (tensor): input feature map
        Returns:
            x (tensor): output feature map
        """

        return self.blocks(x)



class Upsample(nn.Module):

    """
    Upsamling class used in the decoder part
    """

    def __init__(self, scale_factor=None, mode='trilinear'):

        """
        class initializer
        Args:
            scale_factor (int): upsampling scaling factor
            mode (str): upsamling method (trilinear for 3D data)
        Returns:
            None
        """

        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x): 

        """
        forward method
        Args:
            x (tensor): feature map
        Returns:
            upsampled feature maps
        """     

        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        

class PelvicUNet(nn.Module):

    """
    Pelvic U-Net for pelvic OAR segmentations combining deep supervision and shuffle attention modules (SA modules)
    """

    def __init__(self, input_channels, base_num_features, classes, stages, 
                convolutional_blocks_per_stage=2, deep_supervision=True, final_nonlin=sigmoid_activation, 
                pooling_kernels=None, convolutional_kernels=None):

        """
        class initializer
        Args:
            input_channels (int): Number of input channels
            base_num_features (int): output features of the first convolutional layer
            classes (int): number of segmentation classes
            stages (int): stages of the U-Net 
            convolutional_blocks_per_stage (int): convolutional blocks per stage
            deep_supervision (bool): returns segmentation outputs from several stages if True
            final_nonlin (function): sigmoid for multi-label
            pooling_kernels (list): list of ppoling kernels (same length as number of stages)
            convolutional_kernels (list): convolutional kernels (length = stages + 1)
        Returns:
            None
        """

        super().__init__()

        
        # Define layer paramaters
        self.activation_params = {'negative_slope': 1e-2, 'inplace': True}
        self.normalization_params = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        self.convolutional_params = {'stride': 1, 'dilation': 1, 'bias': True}

        self.weightInitializer = InitWeights_He(1e-2)
        self.classes = classes
        self.final_nonlin = final_nonlin
        self.deep_supervision = deep_supervision
  
        
        self.conv_pad_sizes = []
        for krnl in convolutional_kernels:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
        
        self.feature_limit = 320

        self.conv_blocks_encoder = []
        self.conv_blocks_decoder = []
        self.pooling_layers = []
        self.upsampling_layers = []
        self.sa_modules = [] #Shuffle attention modules
        self.seg_outputs = []
        features_out = base_num_features
        features_in = input_channels

        #########################
        # Encoder (contraction) #
        #########################
        for s in range(stages):

            self.convolutional_params['kernel_size'] = convolutional_kernels[s]
            self.convolutional_params['padding'] = self.conv_pad_sizes[s]
            
            # stacked convolutional blocks
            self.conv_blocks_encoder.append(StackedBlock(features_in, features_out, convolutional_blocks_per_stage,
                                                              self.convolutional_params,self.normalization_params, self.activation_params))
            self.pooling_layers.append(nn.MaxPool3d(pooling_kernels[s]))
            features_in = features_out
            features_out = int(np.round(features_out * 2))
           
            features_out = min(features_out, self.feature_limit) #limits features to feature_limit

        ##############
        # Bottleneck #
        ##############
        n_bottleneck_features = self.conv_blocks_encoder[-1].channels_out

        self.convolutional_params['kernel_size'] = convolutional_kernels[stages]
        self.convolutional_params['padding'] = self.conv_pad_sizes[stages]

        
        self.conv_blocks_encoder.append(nn.Sequential(
            StackedBlock(features_in, features_out, convolutional_blocks_per_stage - 1, self.convolutional_params,
                              self.normalization_params, self.activation_params),
            StackedBlock(features_out, n_bottleneck_features, 1, self.convolutional_params,
                              self.normalization_params, self.activation_params)))

        
            
        ##########################
        # Decoder (localization) #
        ########################## 
        for u in range(stages):
            features_from_down = n_bottleneck_features
            features_skip_connection = self.conv_blocks_encoder[-(2 + u)].channels_out 
            features_after_upsampling_and_concat = features_skip_connection * 2

            if u != stages - 1:
                n_bottleneck_features = self.conv_blocks_encoder[-(3 + u)].channels_out
            else:
                n_bottleneck_features = features_skip_connection

            self.upsampling_layers.append(Upsample(scale_factor=pooling_kernels[-(u+1)], mode='trilinear'))

            self.convolutional_params['kernel_size'] = convolutional_kernels[- (u+1)]
            self.convolutional_params['padding'] = self.conv_pad_sizes[- (u+1)]

            self.conv_blocks_decoder.append(nn.Sequential(
                StackedBlock(features_after_upsampling_and_concat, features_skip_connection, convolutional_blocks_per_stage - 1,
                                 self.convolutional_params, self.normalization_params, self.activation_params),
                StackedBlock(features_skip_connection, n_bottleneck_features, 1, self.convolutional_params,
                                  self.normalization_params, self.activation_params)
            ))
            
            self.sa_modules.append(sa_layer(features_from_down))
        
        #gather segmentation outputs of all stages 
        for ds in range(len(self.conv_blocks_decoder)):
            self.seg_outputs.append(nn.Conv3d(self.conv_blocks_decoder[ds][-1].channels_out, classes, 1, 1, 0, 1, 1, False))

        #Module registration
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_decoder)
        self.conv_blocks_encoder = nn.ModuleList(self.conv_blocks_encoder)
        self.pooling_layers = nn.ModuleList(self.pooling_layers)
        self.upsampling_layers = nn.ModuleList(self.upsampling_layers)
        self.sa_modules = nn.ModuleList(self.sa_modules)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

        # weight initialization
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x):

        """
        forward method
        Args:
            x (tensor): model input (usually a 3D patch)
        Returns:
            x (tensor): model ouput (segmentation map)
        """
        
        #initialize empty lists for skip connections and segmentation outputs
        skip_connections = []
        segmentations_out = []

        #Encoder
        for i in range(len(self.conv_blocks_encoder) - 1):
            x = self.conv_blocks_encoder[i](x) #conv block
            skip_connections.append(x) #skip connection
            x = self.pooling_layers[i](x) #maxpooling
        
        #Bottleneck
        x = self.conv_blocks_encoder[-1](x) 

        #Decoder
        for j in range(len(self.upsampling_layers)):
            sa = self.sa_modules[j-(len(self.upsampling_layers))](skip_connections[-(j + 1)]) #SA module takes skip connection as input
            x = self.upsampling_layers[j](x)
            x = torch.cat((x, sa), dim=1) #concat skip connection output and corresponding decoder layer 
            x = self.conv_blocks_decoder[j](x)
            segmentations_out.append(self.final_nonlin(self.seg_outputs[j](x)))

        #get outputs at different levels for deep supervision
        #self.deep_supervision should be True for training, while False for inference
        #if supervision is set to false, only the last network output is returned
        if self.deep_supervision:
            return([segmentations_out[-1]] + [j for j in segmentations_out[:-1][::-1]])
        else:
            return segmentations_out[-1]

  
if __name__ == '__main__':

    from torchsummary import summary

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_channels = 1
    classes = 10
    base_num_features = 32
    stages = 5
    patch_size = (80, 160, 160)
    
    pooling_kernels = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    convolutional_kernels = [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

    model = PelvicUNet(input_channels, base_num_features, classes, stages, deep_supervision = True, pooling_kernels = pooling_kernels,
                    convolutional_kernels=convolutional_kernels, final_nonlin=lambda x: x).to(device)
    
    
    summary(model, (input_channels,) + patch_size)

    x = torch.rand((2,1,80,160,160)).to(device)
    out = model(x)

    