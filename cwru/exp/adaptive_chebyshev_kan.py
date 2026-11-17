"""Reconstructed from the authors' supplementary PDF, pp. 4–5."""
import math
import torch
import torch.nn as nn


class ChebyshevPolynomials(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        T = [torch.ones_like(x), x]
        for n in range(2, self.degree + 1):
            T.append(2 * x * T[n-1] - T[n-2])
        return torch.stack(T, dim=-1)


class AdaptiveChebyshevKANLayer(nn.Module):
    def __init__(self, in_features, out_features, max_degree=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_degree = max_degree
        self.chebyshev = ChebyshevPolynomials(max_degree)
        self.coeffs = nn.Parameter(torch.Tensor(out_features, in_features, max_degree + 1))
        self.order_weights = nn.Parameter(torch.ones(out_features, in_features, max_degree + 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.coeffs, a=math.sqrt(5))

    def forward(self, x):
        x_mapped = torch.tanh(x)
        bases = self.chebyshev(x_mapped)
        gated_coeffs = self.coeffs * torch.sigmoid(self.order_weights)
        return torch.einsum('bid,oid->bo', bases, gated_coeffs)

    def get_symbolic_expression(self, threshold=1e-3):
        expressions = []
        with torch.no_grad():
            effective_coeffs = self.coeffs * torch.sigmoid(self.order_weights)
            for o in range(self.out_features):
                terms = []
                for i in range(self.in_features):
                    for d in range(self.max_degree + 1):
                        c = effective_coeffs[o, i, d].item()
                        if abs(c) > threshold:
                            terms.append(f"{c:.4f} * T_{d}(x_{i})")
                expressions.append(" + ".join(terms) if terms else "0")
        return expressions
