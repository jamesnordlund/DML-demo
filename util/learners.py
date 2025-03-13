from econml.dml import CausalForestDML
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from typing import Optional

from market import Market

class CrossSectionalDML:
    def __init__(self):
        self.fit_model = {}
        self.results = {}
    
    def fit(self, market: Market, time_period: int):
        if time_period in self.fit_model:
            print('Already fitted for this time period')
        dml = CausalForestDML(
            model_y=HistGradientBoostingRegressor(),
            model_t=SampleScaledClassifier(),
            discrete_treatment=True,
            cv=5,
            mc_iters=5,
            mc_agg="median",
            drate=False,
            n_estimators=500,
            criterion="mse",
            max_depth=6,
            min_samples_split=10,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        df = market.data.query(f"time_period == {min(time_period, market.T//2)}")
        dml.fit(
            X = df[market.features],
            Y = market.data.query(f"time_period == {time_period}")["sales"],
            T = market.data.query(f"time_period == {time_period}")["treated"],
        )
        self.fit_model[time_period] = dml
        self.results[time_period] = {
            'ate': dml.ate_inference(X=df[market.features], T0=0, T1=1),
            'ate-1': dml.ate_inference(X=df.query("product_class == 1")[market.features], T0=0, T1=1),
            'ate-4': dml.ate_inference(X=df.query("product_class == 4")[market.features], T0=0, T1=1),
        }
        self.print_fit_model(market, time_period)
        return self

    def print_fit_model(self, market: Market, time_period: int):
        assert time_period in self.fit_model, "Model not fitted for this time period"
        dml = self.fit_model[time_period]
        print("Treatment rate:", market.data["treated"].mean())
        print(f"Nuisance fit diagnostics: treatment: {np.mean(dml.nuisance_scores_t)}, outcome: {np.mean(dml.nuisance_scores_y)}")
        print("Ground truth ATE:", market.data.query(f"time_period == {time_period}")[["sales_control", "sales_treated"]].mean().diff().iloc[-1])
        # print("Ground truth ATT:", market.data.query(f"time_period == {time_period} and treated == 1")[["sales_control", "sales_treated"]].mean().diff().iloc[-1])
        # print("Ground truth ATU:", market.data.query(f"time_period == {time_period} and treated == 0")[["sales_control", "sales_treated"]].mean().diff().iloc[-1])
        print(self.results[time_period]['ate'].summary())

class PlattHistGradientBoostingClassifier(HistGradientBoostingClassifier):
    def __init__(self):
        super().__init__()
        self.platt = LogisticRegression(penalty=None)

    def fit(self, X, y):
        super().fit(X, y)
        self.platt.fit(super().predict_proba(X)[:, [1]], y)
    
    def predict_proba(self, X):
        calibrated_probs = self.platt.predict_proba(super().predict_proba(X)[:, [1]])[:, [1]]
        return np.hstack([1 - calibrated_probs, calibrated_probs])
    
    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return average_precision_score(y, y_pred)

class SampleScaledClassifier(PlattHistGradientBoostingClassifier):
    def __init__(self, beta: Optional[float] = None):
        """Scaling method to adjust propensities for down-sampling.

        Args:
            beta: Scaling factor (e.g. for balanced sampling, this is N_treated / N_control)
        """
        self.beta = beta
        super().__init__()

    def predict_proba(self, X):
        calibrated_probs = super().predict_proba(X)[:, [1]]

        # scale the probabilities by beta
        if self.beta is not None:
            calibrated_probs = (self.beta * calibrated_probs) / (self.beta * calibrated_probs + 1 - calibrated_probs)
        
        return np.hstack([1 - calibrated_probs, calibrated_probs])

    def score(self, X, y):
        if self.beta is not None:
            probs = self.predict_proba(X)[:, 1]
            return brier_score_loss(y, probs)
        return super().score(X, y)