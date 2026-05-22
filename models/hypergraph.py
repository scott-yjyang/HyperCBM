import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg import Hypergraph
from dhg.nn import HGNNPConv


def build_hyperedge(concept_embeddings, k=3):
    """Build hyperedges by selecting top-k most similar concepts per node using cosine similarity."""
    n_concepts = concept_embeddings.shape[0]
    k = min(k, n_concepts - 1) if n_concepts > 1 else 1

    sim_matrix = torch.cosine_similarity(
        concept_embeddings.unsqueeze(1),
        concept_embeddings.unsqueeze(0),
        dim=-1
    )

    diag_idx = torch.arange(n_concepts)
    sim_matrix[diag_idx, diag_idx] = -999.0

    hyperedges = []
    for i in range(n_concepts):
        _, indices = torch.topk(sim_matrix[i], k)
        hyperedges.append(indices.tolist())

    return hyperedges


class HyperEdgeAttention(nn.Module):
    def __init__(self, emb_size, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(emb_size, hidden_dim)
        self.key = nn.Linear(emb_size, hidden_dim)
        self.val = nn.Linear(emb_size, hidden_dim)

    def forward(self, hyperedge_embeddings):
        """
        Args:
            hyperedge_embeddings: [num_hyperedges, k, emb_size]
        Returns:
            attn_weights: [num_hyperedges, k, hidden_dim]
        """
        Q = self.query(hyperedge_embeddings)
        K = self.key(hyperedge_embeddings)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2))
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = attn_scores / (self.hidden_dim ** 0.5)
        attn_weights = torch.matmul(attn_scores, self.val(hyperedge_embeddings))

        return attn_weights


class CrossSampleInteraction(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, batch_embeddings):
        """
        Args:
            batch_embeddings: [batch_size, n_concepts, emb_size]
        Returns:
            global_hyperedges: [num_global_edges, emb_size]
        """
        sample_embeddings = self.global_pool(batch_embeddings)
        num_global_edges = min(5, sample_embeddings.shape[0])
        indices = torch.randperm(sample_embeddings.shape[0])[:num_global_edges]
        return sample_embeddings[indices].squeeze(1)


class HyperConceptNet(nn.Module):
    """HECRL: Hypergraph-Enhanced Concept Representation Learning.

    Builds per-sample kNN hypergraphs over concept embeddings and refines
    them with two layers of hypergraph convolution (HGNNPConv).
    """

    def __init__(self, emb_size, hidden_dim=64):
        super().__init__()
        self.hgcn1 = HGNNPConv(emb_size, hidden_dim)
        self.hgcn2 = HGNNPConv(hidden_dim, emb_size)

    def forward(self, batch_embeddings):
        device = batch_embeddings.device
        batch_size, n_concepts, emb_size = batch_embeddings.shape
        all_outputs = []

        for i in range(batch_size):
            vertex_embeddings = batch_embeddings[i].clone().detach()
            hyperedges = Hypergraph._e_list_from_feature_kNN(vertex_embeddings, k=3)

            hg = Hypergraph(
                num_v=n_concepts,
                e_list=hyperedges,
                device=device
            )

            x = self.hgcn1(batch_embeddings[i], hg)
            x = self.hgcn2(x, hg)
            all_outputs.append(x)

        return torch.stack(all_outputs, dim=0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 10, 32).to(device)
    model = HyperConceptNet(32).to(device)
    output = model(x)
    print(output.shape)
