import numpy as np

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.strategy import FedAvg

class FLMedian(FedAvg):

    def aggregate_fit(self, server_round, results, failures,):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights = [parameters_to_ndarrays(res.parameters)  for _, res in results]
        median = [np.median(np.asarray(layer), axis=0) for layer in zip(*weights)]
        parameters_aggregated = ndarrays_to_parameters(median)

        return parameters_aggregated, {}

    def __init__(
        self,
        *,
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = 2,
        min_evaluate_clients = 2,
        min_available_clients = 2,
        evaluate_fn = None,
        on_fit_config_fn = None,
        on_evaluate_config_fn = None,
        accept_failures = True,
        initial_parameters = None,
        fit_metrics_aggregation_fn = None,
        evaluate_metrics_aggregation_fn = None,
    ):

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    def __repr__(self) -> str:
        rep = f"FLMedian"
        return rep
