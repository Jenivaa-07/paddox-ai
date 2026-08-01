import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerRecommender(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, num_teams=10, num_drivers=20, num_categories=5, num_item_types=3):
        super(TwoTowerRecommender, self).__init__()
        
        # User Tower Embeddings
        self.user_emb = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        self.user_team_emb = nn.Embedding(num_teams + 1, embed_dim // 4, padding_idx=0)
        self.user_driver_emb = nn.Embedding(num_drivers + 1, embed_dim // 4, padding_idx=0)
        
        # Item Tower Embeddings
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.item_type_emb = nn.Embedding(num_item_types + 1, embed_dim // 4, padding_idx=0)
        self.item_cat_emb = nn.Embedding(num_categories + 1, embed_dim // 4, padding_idx=0)
        
        # MLPs
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim + (embed_dim // 4) * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim + (embed_dim // 4) * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward_user(self, user_idx, team_idx, driver_idx):
        u = self.user_emb(user_idx)
        t = self.user_team_emb(team_idx)
        d = self.user_driver_emb(driver_idx)
        x = torch.cat([u, t, d], dim=-1)
        return self.user_mlp(x)
        
    def forward_item(self, item_idx, type_idx, cat_idx):
        i = self.item_emb(item_idx)
        t = self.item_type_emb(type_idx)
        c = self.item_cat_emb(cat_idx)
        x = torch.cat([i, t, c], dim=-1)
        return self.item_mlp(x)
        
    def forward(self, user_idx, team_idx, driver_idx, item_idx, type_idx, cat_idx):
        u_vec = self.forward_user(user_idx, team_idx, driver_idx)
        i_vec = self.forward_item(item_idx, type_idx, cat_idx)
        # Dot product
        score = (u_vec * i_vec).sum(dim=-1)
        return score
