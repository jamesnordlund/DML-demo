import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from typing import Optional

class Market:
    def __init__(
        self,
        N: int,
        T: int,
        fn_lift: callable,
        fn_pull_forward: callable,
        fn_growth: callable,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the market.
        
        Args:
            N: Number of products
            T: number of time periods
            fn_lift: Function specifying the immediate lift effect
            fn_pull_forward: Function specifying the pull forward effect
            fn_growth: Function specifying the growth effect
            seed: Seed for random number generation
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.__make_data__(N, T, fn_lift, fn_pull_forward, fn_growth, **kwargs)

    def __make_data__(
        self,
        N: int,
        T: int,
        fn_lift: callable,
        fn_pull_forward: callable,
        fn_growth: callable,
        **kwargs,
    ) -> None:
        """Create the data for the market.
        
        Args:
            N: Number of products
            T: number of time periods
            fn_lift: Function specifying the immediate lift effect
            fn_pull_forward: Function specifying the pull forward effect
            fn_growth: Function specifying the growth effect
        """
        self.N = N
        self.T = T
        lag_pad = kwargs["lag_pad"] if "lag_pad" in kwargs else T // 4

        self.data = pd.DataFrame(
            {
                "product_id": np.repeat(np.arange(N), T + lag_pad),
                "time_period": np.tile(np.arange(-lag_pad, T), N),
            }
        )

        # generate autocorrelated sales data
        mean = kwargs["mean"] if "mean" in kwargs else 100
        rho = kwargs["rho"] if "rho" in kwargs else 0.95
        sigma = kwargs["sigma"] if "sigma" in kwargs else 30
        col = rho ** np.arange(T+lag_pad) 
        cov_matrix = toeplitz(col) * (sigma**2)
        self.data["sales"] = self.rng.multivariate_normal(mean * np.ones(T+lag_pad), cov_matrix, size=(N,)).flatten()
        
        # generate product classes -- wlog there will be more pull-forward effects in higher-indexed classes
        classes = self.rng.integers(1, 5, size=(N,)) # avoid zero class so lift effect is intuitive
        self.data = self.data.merge(
            right=pd.DataFrame({"product_id": np.arange(N), "product_class": classes}),
            on="product_id"
        )

        # generate product attributes
        K = kwargs["K"] if "K" in kwargs else 5
        attributes = self.rng.normal(5, 1, size=(N,K)) # using positive mean so lift effect is intuitive
        self.data =self.data.merge(
            right=pd.DataFrame(
                data=np.hstack([np.arange(N).reshape(-1,1), attributes]),
                columns=["product_id"] + [f"attribute_{k}" for k in range(K)]
            ),
            on="product_id"
        )

        # store list of features
        self.features = [f"attribute_{k}" for k in range(K)] \
            + ["product_class"] \
            + [f for f in self.data.columns if f.startswith("sales_lag")]

        # update sales given observed features
        features_no_sales = [f for f in self.features if not f.startswith("sales_lag")]
        beta = self.rng.normal(self.rng.uniform(2, 3), 1, size=(len(features_no_sales),))
        self.data["sales"] = self.data["sales"] + self.data[features_no_sales].values @ beta

        # generate treatement assignment propensity as a function of features
        data = self.data.query(f"time_period == {T//2}")
        propensity = 1 / (1 + np.exp(2 - data[features_no_sales].values @ self.rng.normal(0, 0.5, size=(len(features_no_sales),))))
        treated = self.rng.binomial(1, propensity)

        # create the treatment effect
        self.data["sales_control"] = self.data["sales"]
        self.data["treated"] = 1 # quick override to make all products treated
        self.data = fn_lift(self.data, T//2)
        self.data = fn_pull_forward(self.data, T//2)
        self.data = fn_growth(self.data, T//2)
        self.data["sales_treated"] = self.data["sales"] # copy treated sales to a new column
        self.data = self.data.drop(columns=["treated"]).merge(
            right=pd.DataFrame({"product_id": np.arange(N), "treated": treated}),
            on="product_id"
        )
        self.data["sales"] = self.data["sales_control"] * (1 - self.data["treated"]) + self.data["sales_treated"] * self.data["treated"]
        self.data["ite"] = self.data["sales_treated"] - self.data["sales_control"]

        # get lags of sales
        self.data = self.data.sort_values(["product_id", "time_period"])
        for i in range(lag_pad):
            self.data[f"sales_lag_{i}"] = self.data.groupby("product_id")["sales"].shift(i)
        self.data = self.data.query("time_period >= 0")