from config import img_height, img_width, num_classes_exp, device
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import List, Tuple, Optional

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for PyTorch
    """
    def __init__(self, channels: int, ratio: int = 8, name: Optional[str] = None):
        super(SEBlock, self).__init__()
        self.name = name
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // ratio, bias=False)
        self.fc2 = nn.Linear(channels // ratio, channels, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.gap(x).view(b, c)
        
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        
        # Scale
        return x * y.expand_as(x)

class H(nn.Module):
    def __init__(self, in_channels: int, num_filters: int, dropout_rate: float, use_se: bool):
        super(H, self).__init__()
        self.use_se = use_se
        self.num_filters = num_filters
        self.in_channels = in_channels
        
        # BatchNorm para a entrada
        self.bn_input = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Definindo os caminhos de forma mais simples
        self.paths = nn.ModuleList([
            # Path 1: 1x1
            nn.Sequential(
                nn.Conv2d(in_channels, num_filters, kernel_size=1, padding='same', bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True)
            ),
            # Path 2: 3x3
            nn.Sequential(
                nn.Conv2d(in_channels, num_filters, kernel_size=1, padding='same', bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding='same', groups=num_filters, bias=False),
                nn.Conv2d(num_filters, num_filters, kernel_size=1, padding='same', bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True)
            ),
            # Path 3: 5x5
            nn.Sequential(
                nn.Conv2d(in_channels, num_filters, kernel_size=1, padding='same', bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_filters, num_filters, kernel_size=5, padding='same', groups=num_filters, bias=False),
                nn.Conv2d(num_filters, num_filters, kernel_size=1, padding='same', bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True)
            ),
            # Path 4: Pooling
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, num_filters, kernel_size=1, padding='same', bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True)
            )
        ])
        
        # BatchNorm para a saída concatenada
        self.bn_output = nn.BatchNorm2d(4 * num_filters)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        if self.use_se:
            self.se = SEBlock(4 * num_filters, ratio=8)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.relu(x)
        
        # Processar todos os caminhos
        outputs = [path(x) for path in self.paths]
        
        # Concatenar ao longo da dimensão de canais
        x = torch.cat(outputs, dim=1)
        x = self.bn_output(x)
        x = self.relu(x)
        
        if self.use_se:
            x = self.se(x)
            
        return self.dropout(x)
    
class Transition(nn.Module):
    def __init__(self, in_channels: int, compression_factor: float, dropout_rate: float, use_se: bool):
        super(Transition, self).__init__()
        self.use_se = use_se
        
        # BatchNorm matches input channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Calculate output channels with compression
        out_channels = int(np.floor(in_channels * compression_factor))
        
        # 1x1 convolution with compression
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(dropout_rate)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        if self.use_se:
            self.se = SEBlock(out_channels, ratio=8)
            
    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        
        if self.use_se:
            x = self.se(x)
            
        return self.avgpool(x)

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, num_filters: int, 
                 growth_rate: int, dropout_rate: float, use_se_in_H: bool):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Calcula os canais de entrada para esta camada
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(
                H(layer_in_channels, num_filters, dropout_rate, use_se_in_H)
            )
            # Cada camada H adiciona 4*num_filters canais (devido à concatenação)
            growth_rate = 4 * num_filters
            
    def forward(self, x):
        features = [x]
        
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
            
        # Retorna todos os features concatenados e o número total de canais
        return torch.cat(features, dim=1), features[-1].size(1)

class DenseNet(nn.Module):
    def __init__(self, input_shape, num_blocks, num_layers_per_block, growth_rate,
                 dropout_rate, compress_factor, num_filters, num_classes, se_config):
        super(DenseNet, self).__init__()
        
        # Configurações SE
        self.use_se_in_H = 'H' in se_config
        self.use_se_in_transition = 'transicao' in se_config
        self.use_se_in_final = 'topo' in se_config
        
        # Camada inicial
        self.conv1 = nn.Conv2d(input_shape[0], num_filters, kernel_size=3, padding='same', bias=False)
        current_channels = num_filters
        
        # Blocos Densos e Transições
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i in range(num_blocks):
            # Bloco Denso
            dense_block = DenseBlock(
                num_layers_per_block, current_channels, num_filters,
                growth_rate, dropout_rate, self.use_se_in_H
            )
            self.dense_blocks.append(dense_block)
            
            # Atualiza os canais após o bloco denso
            current_channels += num_layers_per_block * (4 * num_filters)
            
            # Transição (exceto após o último bloco)
            if i < num_blocks - 1:
                transition = Transition(
                    current_channels,
                    compress_factor, dropout_rate, self.use_se_in_transition
                )
                self.transitions.append(transition)
                current_channels = int(current_channels * compress_factor)
        
        # SE final se necessário
        if self.use_se_in_final:
            self.final_se = SEBlock(current_channels, ratio=8)
        
        # Camadas finais
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(current_channels, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Adicione verificações de dimensão para depuração
        # print(f"Input shape: {x.shape}")
        
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        
        for i, block in enumerate(self.dense_blocks):
            x, _ = block(x)
            # print(f"After dense block {i+1}: {x.shape}")
            
            if i < len(self.transitions):
                x = self.transitions[i](x)
                # print(f"After transition {i+1}: {x.shape}")
        
        if hasattr(self, 'final_se'):
            x = self.final_se(x)
            # print(f"After final SE: {x.shape}")
        
        x = self.global_avg_pool(x)
        # print(f"After global pool: {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"After flatten: {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"After fc1: {x.shape}")
        x = self.fc2(x)
        # print(f"After fc2: {x.shape}")
        
        return F.softmax(x, dim=1)