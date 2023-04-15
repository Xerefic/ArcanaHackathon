from utils import *

class GConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, num_layers, kernel_size, use_bias):
        super(GConvLSTM, self).__init__()
        
        self.norm = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2, bias=use_bias)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=out_channels, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x = self.norm(x)
        out = self.conv(x)
        out = out.permute(0, 2, 1)
        _, (out, _) = self.lstm(out)
        out = out[-1]
        return out
    
class CrossAttention(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, hidden_dim, num_heads, use_bias=False):
        super(CrossAttention, self).__init__()
        self.heads = num_heads

        self.linear_q = nn.Linear(in_channels_1, hidden_dim*num_heads, bias=use_bias)
        self.linear_k = nn.Linear(in_channels_2, hidden_dim*num_heads, bias=use_bias)
        self.linear_v = nn.Linear(in_channels_2, hidden_dim*num_heads, bias=use_bias)

        self.ff1 = nn.Linear(hidden_dim*num_heads, 4*hidden_dim*num_heads)
        self.ff2 = nn.Linear(4*hidden_dim*num_heads, out_channels)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        q = self.linear_q(x)
        q = q.view(q.size(0), q.size(1)//self.heads, self.heads).moveaxis(-1, 1)
        
        k = self.linear_k(t)
        k = k.view(k.size(0), k.size(1)//self.heads, self.heads)

        v = self.linear_v(t)
        v = v.view(v.size(0), v.size(1)//self.heads, self.heads).moveaxis(-1, 1)

        out = F.softmax(torch.matmul(q, k)/np.sqrt(k.size(1)), dim=-1)
        out = torch.matmul(out, v).flatten(1, -1)

        out = self.ff1(out)
        out = self.relu(out)
        out = self.ff2(out)

        return out

class RiskPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, projected_dim, hidden_dim, num_layers, num_heads, kernel_size, use_bias):
        super(RiskPredictor, self).__init__()

        self.encoder = GConvLSTM(in_channels, projected_dim, hidden_dim, num_layers, kernel_size, use_bias)
        self.cross_attention = CrossAttention(in_channels_1=projected_dim, in_channels_2=1536, out_channels=out_channels, hidden_dim=hidden_dim, num_heads=num_heads)

        self.linear = nn.Linear(out_channels, 1)
    
    def forward(self, x, t):
        out = self.encoder(x)
        out = self.cross_attention(out, t)
        out = self.linear(out)
        return out