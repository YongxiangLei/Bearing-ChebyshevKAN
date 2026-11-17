import torch
import torch.nn as nn
import math

class ChebyshevPolynomials(nn.Module):
    """
    Computes Chebyshev polynomials of the first kind up to degree N.
    """
    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        # x shape: (..., )
        # output shape: (..., degree+1)
        T = [torch.ones_like(x), x]
        for n in range(2, self.degree + 1):
            T.append(2 * x * T[n-1] - T[n-2])
        return torch.stack(T, dim=-1)

class AdaptiveChebyshevKANLayer(nn.Module):
    """
    An adaptive Kolmogorov-Arnold Network layer using Chebyshev polynomials.
    This layer introduces an 'order attention' mechanism that allows the network
    to dynamically weight polynomials of different degrees. Low-frequency signals
    will naturally attend to low-degree polynomials, while high-frequency noise/impulses
    will leverage high-degree polynomials.
    """
    def __init__(self, in_features, out_features, max_degree=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_degree = max_degree

        # Base Chebyshev polynomial generator
        self.chebyshev = ChebyshevPolynomials(max_degree)

        # Learnable coefficients for the polynomials
        # Shape: (out_features, in_features, max_degree + 1)
        self.coeffs = nn.Parameter(torch.Tensor(out_features, in_features, max_degree + 1))

        # Adaptive Order weights: allows turning off/on certain degrees
        # Shape: (out_features, in_features, max_degree + 1)
        self.order_weights = nn.Parameter(torch.ones(out_features, in_features, max_degree + 1))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize coefficients using Xavier-like initialization
        nn.init.kaiming_uniform_(self.coeffs, a=math.sqrt(5))

    def forward(self, x):
        # x shape: (batch_size, in_features)

        # Map x to [-1, 1] for Chebyshev domain stability
        # A common trick is using tanh
        x_mapped = torch.tanh(x)

        # Compute Chebyshev bases
        # bases shape: (batch_size, in_features, max_degree + 1)
        bases = self.chebyshev(x_mapped)

        # Apply adaptive order gating via softmax or sigmoid
        # Here we use a scaled sigmoid to allow soft-selection of orders
        gated_coeffs = self.coeffs * torch.sigmoid(self.order_weights)

        # Multiply bases by coefficients and sum over max_degree and in_features
        # bases: (B, I, D)
        # gated_coeffs: (O, I, D)
        # We want output: (B, O)
        # Einsum notation:
        # b: batch, i: in_features, d: degree, o: out_features
        out = torch.einsum('bid,oid->bo', bases, gated_coeffs)

        return out

    def get_symbolic_expression(self, threshold=1e-3):
        """
        Extracts a symbolic representation by filtering out small coefficients.
        This provides the "White-Box" interpretability required for AAAI.
        """
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

# Example Usage
if __name__ == "__main__":
    layer = AdaptiveChebyshevKANLayer(in_features=128, out_features=64, max_degree=8)
    dummy_input = torch.randn(32, 128)  # 32 batch size, 128 features (e.g. signal segment)
    output = layer(dummy_input)
    print(f"Output shape: {output.shape}")

    # Demonstrate Symbolic Extraction
    # In practice, this would be called after training to show physical interpretability
    print("\nExtracted Symbolic Expressions for first 2 output neurons:")
    exprs = layer.get_symbolic_expression(threshold=0.5)
    for idx in range(2):
        print(f"y_{idx} = {exprs[idx][:200]}...") # truncate for display
