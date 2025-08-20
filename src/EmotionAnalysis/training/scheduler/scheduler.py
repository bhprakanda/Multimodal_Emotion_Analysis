import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau


class SchedulerFactory:
    @staticmethod
    def get_scheduler(
        name: str,
        optimizer: Optimizer,
        *,
        mode: str = 'max',
        factor: float = 0.1,
        patience: int = 10,
        verbose: bool = True,
        total_steps: int = None,
        max_lr: float = None,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 100.0,
        anneal_strategy: str = 'cos',
        cycle_momentum: bool = False
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns a configured learning rate scheduler based on name.

        Parameters:
        - name (str): Scheduler name ("onecycle" or "reducelronplateau").
        - optimizer (Optimizer): Optimizer to apply the scheduler to.
        - mode, factor, patience, verbose: Used for ReduceLROnPlateau.
        - total_steps, max_lr, pct_start, etc.: Used for OneCycleLR.

        Returns:
        - torch.optim.lr_scheduler._LRScheduler: Scheduler object.
        """

        name = name.lower()

        if name == "onecycle":
            if not all([max_lr, total_steps]):
                raise ValueError("OneCycleLR requires 'max_lr' and 'total_steps'.")
            return OneCycleLR(
                optimizer=optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                anneal_strategy=anneal_strategy,
                cycle_momentum=cycle_momentum,
                verbose=verbose
            )

        elif name == "reducelronplateau":
            return ReduceLROnPlateau(
                optimizer=optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                verbose=verbose
            )

        else:
            raise ValueError(f"Unsupported scheduler name: {name}")


# if __name__ == "__main__":
    
#     from scheduler import SchedulerFactory

#     # For ReduceLROnPlateau (factor=0.75, patience=4)
#     scheduler = SchedulerFactory.get_scheduler(
#         name="reducelronplateau",
#         optimizer=optimizer,
#         mode='max',
#         factor=0.75,
#         patience=4,
#         verbose=True
#     )


#     # For OneCycleLR
#     scheduler = SchedulerFactory.get_scheduler(
#         name="onecycle",
#         optimizer=optimizer,
#         max_lr=Config.max_lr,
#         total_steps=total_steps,
#         pct_start=0.3,
#         div_factor=25,
#         final_div_factor=100,
#         anneal_strategy='cos',
#         cycle_momentum=False  # for AdamW
#     )


#     # For ReduceLROnPlateau (factor=0.5, patience=5)
#     scheduler = SchedulerFactory.get_scheduler(
#         name="reducelronplateau",
#         optimizer=optimizer,
#         mode='max',
#         factor=0.5,
#         patience=5,
#         verbose=True
#     )