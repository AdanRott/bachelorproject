""" This code was adjusted from Professor Andrej Karpathy's course in Neural Networks https://karpathy.ai/zero-to-hero.html"""

class GraphHead(nn.Module):
    """ single head unit according to the graph attentional paper modified to include edge_features code was adapted from Karpathy"""
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, x, adj_matrix, current_batch_size=BATCH_SIZE, non_linearity=True):
        x = x.view(current_batch_size, -1, EMBED_DIM)
        adj_matrix = adj_matrix.view(current_batch_size, -1, MAX_X_SIZE)

        key = self.key(x)   
        query = self.query(x) 

        # Compute the attention weights
        wei = query @ key.transpose(-2,-1) * (EMBED_DIM ** -0.5
        wei = wei * adj_matrix
        mask = (adj_matrix == 0)

        # Mask out the weights of the non-existent edges
        wei = wei.masked_fill(mask, float('-inf'))
        wei = F.softmax(wei, dim=-1) 

        mask = torch.isnan(wei)
        wei = torch.where(mask, torch.zeros_like(wei), wei)

        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x) 

        out = wei @ v 

        # Introduce non-linearity
        if non_linearity:
          out = self.activation(out)

        return out

class MultiGraphHead(nn.Module):
    """ Multiple head unit according to Attention is all you need paper"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.graph_heads = nn.ModuleList([GraphHead(head_size) for _ in range(num_heads)])
        self.linearlayer = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, x, adj_matrix, current_batch_size=BATCH_SIZE):
    
        # Concatenate the outputs of all the attention heads
        x = torch.cat([f(x, adj_matrix, current_batch_size) for f in self.graph_heads], dim=-1)
        out = self.dropout(self.linearlayer(x))
        return out

class FeedForwardlayer(nn.Module):
    """ Feed forward layer according to Attention is all you need paper """
    def __init__(self):
          super().__init__()
          self.linear1 = nn.Linear(EMBED_DIM, 4 * EMBED_DIM)
          self.relu = nn.ReLU()
          self.linear2 = nn.Linear(4 * EMBED_DIM, EMBED_DIM)
          self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, x):
          x = self.linear1(x)
          x = self.relu(x)
          x = self.linear2(x)
          x = self.dropout(x)
          return x

class GraphTransformerBlock(nn.Module):
    """ Combines the multihead and feedforward layer"""
    def __init__(self, num_heads):
        super().__init__()
        head_size = EMBED_DIM // num_heads
        self.multihead = MultiGraphHead(num_heads, head_size)
        self.feedforwardlayer = FeedForwardlayer()
        self.layernorm1 = nn.LayerNorm(EMBED_DIM)
        self.layernorm2 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x, adj_matrix, current_batch_size=BATCH_SIZE):
    
        # Residual connections and layer normalizations
        x =  x.view(current_batch_size, -1, EMBED_DIM) + self.multihead(self.layernorm1(x), adj_matrix, current_batch_size)
        x =  x + self.feedforwardlayer(self.layernorm2(x))
        return x