import torch.nn as nn
import torch.nn.functional as F

class ReviewClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.5):
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_sigmoid=False):
        intermediate = F.relu(self.batch_norm1(self.fc1(x_in)))
        intermediate = self.dropout(intermediate)
        output = self.fc2(intermediate).squeeze()
        if apply_sigmoid:
            output = F.sigmoid(output)
        return output