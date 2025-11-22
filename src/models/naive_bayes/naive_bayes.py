import numpy as np

class NaiveBayes:
    def __init__(self):
        # Will be filled during fit()
        self.class_priors = {}
        self.gaussian_params = {}      # mean & var for continuous features
        self.categorical_params = {}   # prob tables for categorical features
        self.feature_types = []        # "gaussian" or "categorical"

    def fit(self, X, y, feature_types):
        self.feature_types = feature_types
        classes = np.unique(y)
        
        # Compute priors
        for cls in classes:
            self.class_priors[cls] = np.mean(y == cls)

        # Initialize likelihood storage
        for cls in classes:
            self.gaussian_params[cls] = {}
            self.categorical_params[cls] = {}

        # Fit each feature separately
        for feature_idx in range(X.shape[1]):
            ftype = feature_types[feature_idx]

            if ftype == "gaussian":
                # Compute mean and variance for each class
                for cls in classes:
                    values = X[y == cls, feature_idx]
                    mean = values.mean()
                    var = values.var() + 1e-6  # stability
                    self.gaussian_params[cls][feature_idx] = (mean, var)

            elif ftype == "categorical":
                # Count frequency of each category per class
                for cls in classes:
                    values = X[y == cls, feature_idx]
                    unique_vals, counts = np.unique(values, return_counts=True)
                    total = counts.sum()

                    # Laplace smoothing
                    probs = {val: (count + 1) / (total + len(unique_vals))
                             for val, count in zip(unique_vals, counts)}

                    self.categorical_params[cls][feature_idx] = probs

        return self


    def _gaussian_likelihood(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)


    def _categorical_likelihood(self, x, prob_table):
        # If unseen category → Laplace smoothing fallback
        return np.log(prob_table.get(x, 1e-6))


    def predict_log_proba_one(self, x):
        """
        Computes log posterior for ONE sample.
        """
        log_posteriors = {}

        for cls in self.class_priors:
            log_p = np.log(self.class_priors[cls])

            for feature_idx, value in enumerate(x):
                ftype = self.feature_types[feature_idx]

                if ftype == "gaussian":
                    mean, var = self.gaussian_params[cls][feature_idx]
                    log_p += self._gaussian_likelihood(value, mean, var)

                elif ftype == "categorical":
                    probs = self.categorical_params[cls][feature_idx]
                    log_p += self._categorical_likelihood(value, probs)

            log_posteriors[cls] = log_p

        return log_posteriors


    def predict(self, X):
        preds = []
        for x in X:
            log_probs = self.predict_log_proba_one(x)
            preds.append(max(log_probs, key=log_probs.get))
        return np.array(preds)


    def predict_proba(self, X):
        all_probs = []
        for x in X:
            log_probs = self.predict_log_proba_one(x)

            # convert log probs → normalized probabilities
            logs = np.array(list(log_probs.values()))
            exp_logs = np.exp(logs - logs.max())   # stability
            probs = exp_logs / exp_logs.sum()

            all_probs.append(probs)
        return np.array(all_probs)