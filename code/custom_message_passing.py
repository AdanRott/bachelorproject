class CustomMessagePassing(nn.Module):
    """This function takes in the edge attributes, edge indices, atom features a
    and length parameters and returns the updated
    atom features."""
    def __init__(self, in_edge_features, out_edge_features, in_atom_features):
        super(CustomMessagePassing, self).__init__()

        self.edge_transform = nn.Linear(in_edge_features, out_edge_features)
        self.aggr_transform = nn.Linear(out_edge_features, out_edge_features, bias=False)
        self.atom_transform = nn.Linear(in_atom_features, out_edge_features)
        self.feature_dim = out_edge_features

    def forward(self, edge_index, atom_features, edge_attr, atom_features_length, edge_attr_length):
        
        current_batch_size = atom_features_length.shape[0]

        # Create masks based on length parameters
        edge_mask = (torch.arange(edge_attr.size(1)).to(device) < edge_attr_length).float().unsqueeze(-1)
        atom_mask = (torch.arange(atom_features.size(1)).to(device) < atom_features_length).float()

        # Transform edge attributes
        extended_edge_attr = self.edge_transform(edge_attr)
        extended_edge_attr = torch.nn.LeakyReLU(0.2)(extended_edge_attr)

        # Masking edge attributes
        edge_mask_expanded = edge_mask.view(current_batch_size, -1, 1).expand_as(extended_edge_attr)
        masked_extended_edge_attr = extended_edge_attr * edge_mask_expanded

        # Transform atom_features
        atom_features_transformed = self.atom_transform(atom_features)
        atom_features_transformed = torch.nn.LeakyReLU(0.2)(atom_features_transformed)

        # Masking atom features
        atom_mask_expanded = atom_mask.view(current_batch_size, -1, 1).expand_as(atom_features_transformed)
        masked_atom_features_transformed = atom_features_transformed * atom_mask_expanded

        edge_index = edge_index % 42
        src_nodes = edge_index[:, 0, :]
        dest_nodes = edge_index[:, 1, :]

        # Initialize aggregated neighbors tensor
        aggregated_neighbors = torch.zeros_like(atom_features_transformed)

        # Build aggregated neighbors tensor
        flat_masked = masked_atom_features_transformed.view(current_batch_size * 42, -1)
        adjusted_src_nodes = src_nodes + torch.arange(0, current_batch_size * 42, 42).view(-1, 1).to(device)

        dest_nodes = dest_nodes.unsqueeze(-1).expand(-1, -1, self.feature_dim)
        aggregated_neighbors.scatter_add_(1, dest_nodes, masked_extended_edge_attr * flat_masked[adjusted_src_nodes])

        # Apply learnable weights to aggregated neighbors
        aggregated_neighbors = self.aggr_transform(aggregated_neighbors)
        aggregated_neighbors = torch.tanh(aggregated_neighbors)

        # Add original atom features
        output = aggregated_neighbors + masked_atom_features_transformed

        return output, extended_edge_attr