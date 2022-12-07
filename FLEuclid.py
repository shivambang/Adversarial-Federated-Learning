import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.strategy import FedAvg

class FLEuclid(FedAvg):

    def aggregate_fit(self, server_round, results, failures,):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights = [parameters_to_ndarrays(res.parameters)  for _, res in results]
        deltas = []
        for i in range(len(weights)):
            deltas.append(np.asarray(weights[i][-2]) - np.asarray(self.current_weights[-2]))
        deltas = np.reshape(np.asarray(deltas), (len(weights), -1))
        print(deltas.shape)
        def get_euclid_scores(deltas, size):

            euclid_scores = np.zeros(len(deltas))
            distances = pairwise_distances(deltas)

            for i in range(len(deltas)):
                euclid_scores[i] = np.sum(np.sort(distances[i])[1:(size - 1)])

            return euclid_scores
        
        scores = get_euclid_scores(deltas, len(deltas) - 2)
        good_idx = np.argpartition(scores, len(deltas) - 2)[:(len(deltas) - 2)]
        weights = [np.mean(np.asarray(layer)[good_idx], axis=0) for layer in zip(*weights)]
        parameters_aggregated = ndarrays_to_parameters(weights)

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
        initial_parameters,
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
        self.current_weights = parameters_to_ndarrays(initial_parameters)

    def __repr__(self) -> str:
        rep = f"FLEuclid"
        return rep
