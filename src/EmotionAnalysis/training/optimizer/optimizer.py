import torch
from torch import nn
from torch.optim import Optimizer


class OptimizerFactory:
    @staticmethod
    def get_optimizer(
        name: str,
        model: nn.Module,
        lr: float,
        weight_decay: float = 0.0,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8
    ) -> Optimizer:
        """
        Returns a configured optimizer.

        Parameters:
        - name (str): Name of the optimizer (e.g., "adamw").
        - model (nn.Module): Model whose parameters will be optimized.
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay (L2 regularization).
        - betas (tuple): Betas for Adam-based optimizers.
        - eps (float): Epsilon for numerical stability.

        Returns:
        - Optimizer: A PyTorch optimizer.
        """
        name = name.lower()

        if name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {name}")
        

# if __name__ == "__main__":

#     from optimizer import OptimizerFactory

#     # Example 1
#     optimizer = OptimizerFactory.get_optimizer(
#         name="adamw",
#         model=model,
#         lr=Config.base_lr,
#         weight_decay=Config.weight_decay,
#         betas=(0.9, 0.999),
#         eps=1e-8
#     )

#     # Example 2
#     optimizer = OptimizerFactory.get_optimizer(
#     name="adamw",
#     model=model,
#     lr=config["learning_rate"],
#     weight_decay=config["weight_decay"]
#     )

#     # Example 2
#     optimizer = OptimizerFactory.get_optimizer(
#     name="adamw",
#     model=model,
#     lr=5e-5,
#     weight_decay=1e-3
#     )