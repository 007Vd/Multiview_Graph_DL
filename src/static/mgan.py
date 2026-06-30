#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.models.LSTM_attn.encoder import StockEncoder
class HistoricalState(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.W1 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Ve = nn.Parameter(torch.randn(hidden_dim)* 0.1)
        self.scale = hidden_dim ** 0.5

    def forward(self, H, X_last):
        B_total, L, Hdim = H.shape
        H_last = H[:, -1, :].unsqueeze(1).repeat(1, L, 1)
        concat = torch.cat([H, H_last], dim=-1)
        term1 = self.W1(concat)
        term2 = self.W2(X_last).unsqueeze(1)
        e = torch.tanh(term1 + term2)
        scores = torch.matmul(e, self.Ve)
        alpha = torch.softmax(scores, dim=1)
        E = (alpha.unsqueeze(-1) * H).sum(dim=1)
        return E, alpha
    

class StockEncoder(nn.Module):
    def __init__(self, num_assets, hidden_dim=64):
        super().__init__()
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.attn = HistoricalState(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        B = x.shape[0]
        x = x.permute(0, 2, 1)
        x = x.reshape(B * self.num_assets, -1, 1)
        H, _ = self.lstm(x) 
        X_last = x[:, -1, :]
        E, alpha = self.attn(H, X_last) 
        E = E.view(B, self.num_assets, self.hidden_dim)
        return self.norm(E), alpha

# class TGAHead(nn.Module):
#     def __init__(self, hidden_dim, adjacency_matrix):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         adj = adjacency_matrix.float()
#         adj = (adj + torch.eye(adj.shape[0])).clamp(max=1)        # self-loops
#         self.register_buffer("adj", adj)
#         self.num_nodes = adj.shape[0]
#         self.W        = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.a        = nn.Linear(2 * hidden_dim, 1, bias=False)
#         self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

#     def forward(self, E):
#         B, N, D = E.shape
#         H = self.W(E)                                             # (B, N, D)

#         Hv = H.unsqueeze(2).expand(-1, -1, N, -1)                # (B, N, N, D)
#         Hu = H.unsqueeze(1).expand(-1, N, -1, -1)                # (B, N, N, D)

#         scores = self.a(torch.cat([Hu, Hv], dim=-1)).squeeze(-1)  # (B, N, N)

#     # Soft mask: multiply by adj instead of -inf
#     # Keeps gradients flowing through self.a for sparse graphs
#         scores = scores * self.adj.unsqueeze(0)                   # (B, N, N)

#         alpha  = F.softmax(scores, dim=2)                         # (B, N, N)
#         output = torch.bmm(alpha, H)                              # (B, N, D)
#         output = self.out_proj(output)
#         return output, alpha

class TGAHead(nn.Module):
    def __init__(self, hidden_dim, adjacency_matrix):
        super().__init__()
        self.hidden_dim = hidden_dim
        adj = adjacency_matrix.float()
        adj = (adj + torch.eye(adj.shape[0])).clamp(max=1)
        self.register_buffer("adj", adj)
        self.num_nodes = adj.shape[0]
        self.W        = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.a        = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, E):
        B, N, D = E.shape
        H = self.W(E)

        Hv = H.unsqueeze(2).expand(-1, -1, N, -1)
        Hu = H.unsqueeze(1).expand(-1, N, -1, -1)

        scores = self.a(torch.cat([Hu, Hv], dim=-1)).squeeze(-1)
        scores = scores * self.adj.unsqueeze(0)
        alpha  = F.softmax(scores, dim=2)
        output = torch.bmm(alpha, H)
        output = self.out_proj(output)
        return output, alpha
# %%
class MGANLayer(nn.Module):
    def __init__(self,hidden_dim,adjacency_matrix,num_heads=4):
        super().__init__()
        self.heads=nn.ModuleList(
            [TGAHead(hidden_dim,adjacency_matrix)
             for _ in range(num_heads)])
        self.fusion=nn.Linear(num_heads*hidden_dim,hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self,E):
        outputs=[]
        attentions=[]
        for head in self.heads:
            out,alpha=head(E)
            outputs.append(out)
            attentions.append(alpha)
        
        multihead=torch.cat(outputs,dim=-1)
        embeddings=self.fusion(multihead)
        return self.norm(embeddings + E), attentions
    
# %%

class PortfolioHead(nn.Module):
    def __init__(self,hidden_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim * 4)
        self.network=nn.Sequential(
            nn.Linear(hidden_dim*4,128),
            nn.SiLU(),

            nn.Linear(128,64),
            nn.SiLU(),

            nn.Linear(64,1)
        )

    def forward(self,x):
        x = self.norm(x)    
        scores=self.network(x).squeeze(-1)
        weights=F.softmax(scores,dim=1)

        return weights
    

sector_adj=torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_sector.pt")
industry_adj=torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_industry.pt")
theme_adj=torch.load("/Users/007vd/Downloads/DAU/Split_Graph_dl/data/static/A_theme.pt")


class MGANPortfolio(nn.Module):
    def __init__(self,sector_adj,industry_adj,theme_adj,num_assets=15,hidden_dim=64,num_heads=4):
        super().__init__()
        self.encoder=StockEncoder(num_assets=num_assets,hidden_dim=hidden_dim)
        
        self.sector_graph=MGANLayer(hidden_dim=hidden_dim,adjacency_matrix=sector_adj,num_heads=num_heads)
        self.industry_graph=MGANLayer(hidden_dim=hidden_dim,adjacency_matrix=industry_adj,num_heads=num_heads)
        self.theme_graph=MGANLayer(hidden_dim=hidden_dim,adjacency_matrix=theme_adj,num_heads=num_heads)

        self.portfolio_head=PortfolioHead(hidden_dim=hidden_dim)

    def forward(self,x):
        E_stock,stock_attention=self.encoder(x)
        E_sector, sector_attention = self.sector_graph(E_stock)
        E_industry, industry_attention = self.industry_graph(E_sector)
        E_theme, theme_attention = self.theme_graph(E_industry)

        fused_embeddings=torch.cat([
            E_stock,E_sector,E_industry,E_theme
        ],dim=-1)

        weights=self.portfolio_head(fused_embeddings)

        return weights
    
      