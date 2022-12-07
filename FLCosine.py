import torch
from functools import reduce
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.strategy import FedAvg

class FLCosine(FedAvg):

    def aggregate_fit(self, server_round, results, failures,):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights = [parameters_to_ndarrays(res.parameters)  for _, res in results]

        def cos_score(grads):
            cos_sim = cosine_similarity(np.reshape(np.asarray(grads), (len(grads), -1))) - np.eye(len(grads))
            max_cos_sim = np.max(cos_sim, axis=1)
            for i in range(len(grads)):
                for j in range(len(grads)):
                    if i == j:
                        continue
                    if max_cos_sim[i] < max_cos_sim[j]:
                        cos_sim[i][j] = cos_sim[i][j] * max_cos_sim[i] / max_cos_sim[j]
            w = 1 - (np.max(cos_sim, axis=1))
            w[w > 1] = 1
            w[w < 0] = 0

            w = w / np.max(w)
            w[(w == 1)] = .99
            
            # Logit function
            w = (np.log(w / (1 - w)) + 0.5)
            w[(np.isinf(w) + w > 1)] = 1
            w[(w < 0)] = 0

            return w    
        grads = []       
        for i in range(len(weights)):
            grads.append(np.asarray(weights[i][-2]) - np.asarray(self.current_weights[-2]))
        wv = cos_score(grads)
        
        deltas = []
        for i in range(len(weights)):
            deltas.append(np.asarray(weights[i]) - np.asarray(self.current_weights))
        weighted_deltas = [
            deltas[layer] * wv[layer] for layer in range(len(weights))
        ]
        weights_new = np.sum(weighted_deltas, axis=0) / len(weights)
        self.current_weights = np.asarray(self.current_weights) + weights_new
        parameters_aggregated = ndarrays_to_parameters(self.current_weights)

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
        rep = f"FLCosine"
        return rep
