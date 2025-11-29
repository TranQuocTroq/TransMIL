import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):
    """
    Transformer Layer using Nystrom Attention for efficient long-sequence modeling.
    """
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,    # Number of landmarks for Nystrom approximation
            pinv_iterations=6,         # Number of Moore-Penrose iterations
            residual=True,             # Apply residual connection
            dropout=0.1
        )

    def forward(self, x):
        """
        Forward pass: Normalization -> Nystrom Attention -> Residual Connection
        """
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    """
    Position Encoding Generator (PPEG).
    Introduces conditional position encoding using depth-wise separable convolutions.
    """
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        
        # Reshape sequence back to 2D spatial grid for convolution
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        
        # Apply multi-scale convolutions
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        
        # Flatten back to sequence
        x = x.flatten(2).transpose(1, 2)
        
        # Concatenate with CLS token
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    """
    TransMIL: Transformer based Correlated Multiple Instance Learning.
    Modified to support variable input dimensions (e.g., 512 for CONCH, 1024 for ResNet).
    """
    def __init__(self, n_classes, input_dim=512):
        """
        Args:
            n_classes (int): Number of target classes.
            input_dim (int): Dimension of input features (default: 512).
        """
        super(TransMIL, self).__init__()
        
        self.pos_layer = PPEG(dim=512)
        
        # Linear projection: Input Dimension -> 512
        # Adjusted from hardcoded 1024 to dynamic input_dim
        self._fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        
        # Learnable Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        
        # Classification Head
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, **kwargs):
        """
        Forward pass of the model.
        Args:
            kwargs: Dictionary containing 'data' (tensor of shape [B, N, input_dim])
        """
        # Handle input data from dictionary (standard MIL interface)
        # Use 'data' key if available, otherwise fallback to 'x'
        if 'data' in kwargs:
            h = kwargs['data'].float()
        elif 'x' in kwargs:
            h = kwargs['x'].float()
        else:
            raise ValueError("Input dictionary must contain key 'data' or 'x'")

        # Project features to 512 dim: [B, n, input_dim] -> [B, n, 512]
        h = self._fc1(h)

        # ----> Padding to Square
        # Determine spatial dimensions for PPEG
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        
        # Pad sequence with existing features to match square size
        h = torch.cat([h, h[:, :add_length, :]], dim=1) 

        # ----> Prepend Class Token
        B = h.shape[0]
        # Expand CLS token to batch size and move to correct device
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        # ----> Transformer Layer 1
        h = self.layer1(h) 

        # ----> Position Encoding (PPEG)
        h = self.pos_layer(h, _H, _W) 

        # ----> Transformer Layer 2
        h = self.layer2(h) 

        # ----> Feature Aggregation
        # Extract the CLS token after processing
        h = self.norm(h)[:, 0]

        # ----> Prediction
        logits = self._fc2(h) # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

if __name__ == "__main__":
    # Test block to verify dimension compatibility
    # Simulating a batch with CONCH features (dim=512)
    input_dim = 512
    num_patches = 6000
    batch_size = 1
    num_classes = 5 # UBC-OCEAN setting
    
    data = torch.randn((batch_size, num_patches, input_dim)).cuda()
    
    # Initialize model with input_dim=512
    model = TransMIL(n_classes=num_classes, input_dim=input_dim).cuda()
    print(f"Model initialized. Input Dim: {input_dim}, Classes: {num_classes}")
    
    model.eval()
    with torch.no_grad():
        results_dict = model(data=data)
    
    print("Output Logits Shape:", results_dict['logits'].shape)
    print("Test passed successfully.")
