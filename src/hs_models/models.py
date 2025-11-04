from typing import Dict, Union
import numpy as np
import pandas as pd
import pymc as pm

from pymc_extras.model_builder import ModelBuilder

class LinearPooledNoInteraction(ModelBuilder):
    # Give the model a name
    _model_type = "LinearPooledNoInteraction"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model for making predictions on new areas

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """

        area_idx_int, self.area_map = pd.factorize(X['area_id'], sort=True)
        self.n_areas = len(self.area_map)

        coords = {
            "area": self.area_map,
            "area_id": area_idx_int
        }

        # Check the type of X and y and adjust access accordingly
        self._generate_and_preprocess_model_data(X, y)

        y = y.values if isinstance(y, pd.Series) else y

        with pm.Model(coords=coords) as self.model:
            # Create data containers
            x_data = pm.Data(
                    "x_data", 
                    X['total_counts'].values,
                )
            y_obs = pm.Data("y_obs", y)

            β1_mu_prior         = self.model_config.get("β1_mu_prior", 0.02)
            β1_sigma_prior      = self.model_config.get("β1_sigma_prior", 0.03)

            σ_prior = self.model_config.get("σ_prior", 600.0)

            # global coefficient priors
            β1 = pm.HalfNormal("β1", sigma=β1_sigma_prior)

            # model error
            σ = pm.HalfCauchy("σ", σ_prior)

            # linear mean 
            mu = β1*x_data

            # likelihood
            obs = pm.Gamma(
                "y", 
                mu=mu, 
                sigma=σ, 
                observed=y_obs, 
            )

    def _data_setter(
        self, 
        X: pd.DataFrame, 
        y: pd.Series= None
    ):
        area_idx_int = self.area_map.get_indexer(X['area_id'])
        
        with self.model:
            pm.set_data({
                "x_data": X.total_counts.values,
            })
            if y is not None:
                pm.set_data({"y_obs": y.values})

    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config: Dict = { 
            "β1_mu_prior": 0.02,
            "β1_sigma_prior": 0.03,
            "σ_prior": 600.0
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: Dict = {
            "draws": 1000,
            "tune": 1000,
            "cores": 4,
            "chains": 4,
            "target_accept": 0.98,
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """

        pass

    def _generate_and_preprocess_model_data(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y


class LinearPartPoolNoInteraction(ModelBuilder):
    # Give the model a name
    _model_type = "LinearPartPoolNoInteraction"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model for making predictions on new areas

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """

        area_idx_int, self.area_map = pd.factorize(X['area_id'], sort=True)
        self.n_areas = len(self.area_map)

        coords = {
            "area": self.area_map,
            "area_id": area_idx_int
        }

        # Check the type of X and y and adjust access accordingly
        self._generate_and_preprocess_model_data(X, y)

        y = y.values if isinstance(y, pd.Series) else y

        with pm.Model(coords=coords) as self.model:
            # Create data containers
            area_idx = pm.Data("area_idx", area_idx_int)

            # Create data containers
            x_data = pm.Data(
                    "x_data", 
                    X['total_counts'].values,
                )
            y_obs = pm.Data("y_obs", y)

            β1_sigma_mu_prior      = self.model_config.get("β1_sigma_mu_prior", 0.03)
            β1_sigma_std_prior      = self.model_config.get("β1_sigma_std_prior", 0.06)

            σ_prior = self.model_config.get("σ_prior", 600.0)

            # prior on the shared distribution of slopes
            sigma_b1 = pm.Gamma("sigma_b", mu=β1_sigma_prior, sigma=β1_sigma_std_prior)

            # group-specific slope priors
            β1 = pm.HalfNormal("β1", sigma=sigma_b1, dims="area")

            # model error
            σ = pm.HalfCauchy("σ", σ_prior)

            # linear mean 
            mu = β1[area_idx]*x_data

            # likelihood
            obs = pm.Gamma(
                "y", 
                mu=mu, 
                sigma=σ, 
                observed=y_obs, 
            )

    def _data_setter(
        self, 
        X: pd.DataFrame, 
        y: pd.Series= None
    ):
        area_idx_int = self.area_map.get_indexer(X['area_id'])
        
        with self.model:
            pm.set_data({
                "x_data": X.total_counts.values,
            })
            if y is not None:
                pm.set_data({"y_obs": y.values})

    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config: Dict = { 
            "β1_sigma_mu_prior": 0.03,
            "β1_sigma_std_prior": 0.06,
            "σ_prior": 600.0
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: Dict = {
            "draws": 1000,
            "tune": 1000,
            "cores": 4,
            "chains": 4,
            "target_accept": 0.98,
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """

        pass

    def _generate_and_preprocess_model_data(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y


class AreaCountInteraction1DPartPool(ModelBuilder):
    # Give the model a name
    _model_type = "AreaCountInteraction1DPartPool"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model for making predictions on new areas

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """

        area_idx_int, self.area_map = pd.factorize(X['area_id'], sort=True)
        self.n_areas = len(self.area_map)

        coords = {
            "area": self.area_map,
        }

        # Check the type of X and y and adjust access accordingly
        self._generate_and_preprocess_model_data(X, y)

        y = y.values if isinstance(y, pd.Series) else y

        with pm.Model(coords=coords) as self.model:
            # Create data containers
            area_idx = pm.Data("area_idx", area_idx_int)

            x_data = pm.Data(
                    "x_data", 
                    X['total_counts'].values,
                )
            m_data = pm.Data("m_data", X['area_size'].values)
            y_obs = pm.Data("y_obs", y)

            β1_mu_prior         = self.model_config.get("β1_mu_prior", 0.0)
            β1_sigma_prior      = self.model_config.get("β1_sigma_prior", 10.0)
            β2_mean_mu_prior    = self.model_config.get("β2_mean_mu_prior", 0.0)
            β2_mean_sigma_prior = self.model_config.get("β2_mean_sigma_prior", 10.0)
            β2_std_lambda_prior = self.model_config.get("β2_std_lambda_prior", 1.0)

            σ_prior = self.model_config.get("σ_prior", 1000.0)

            # global coefficient priors
            β1 = pm.Normal("β1", mu=β1_mu_prior, sigma=β1_sigma_prior)

            # group-specific area-count interaction term priors
            mu_b2 = pm.Normal("mu_b", mu=β2_mean_mu_prior, sigma=β2_mean_sigma_prior)
            sigma_b2 = pm.Exponential("sigma_b", β2_std_lambda_prior)
            β2 = pm.Normal("β2", mu=mu_b2, sigma=sigma_b2, dims="area")

            # model error
            σ = pm.HalfCauchy("σ", σ_prior)

            # linear mean 
            mu = β1*x_data + β2[area_idx]*x_data*m_data

            # likelihood
            obs = pm.Normal(
                "y", 
                mu=mu, 
                sigma=σ, 
                observed=y_obs, 
            )

    def _data_setter(
        self, 
        X: pd.DataFrame, 
        y: pd.Series= None
    ):
        area_idx_int = self.area_map.get_indexer(X['area_id'])
        
        with self.model:
            pm.set_data({
                "x_data": X.total_counts.values,
                "m_data": X.area_size.values,
                "area_idx": area_idx_int,
            })
            if y is not None:
                pm.set_data({"y_obs": y.values})

    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config: Dict = { 
            "β1_mu_prior": 0.0,
            "β1_sigma_prior": 10.0,
            "β2_mean_mu_prior": 0.0,
            "β2_mean_sigma_prior": 10.0,
            "β2_std_lambda_prior": 1.0,
            "σ_prior": 1000.0
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: Dict = {
            "draws": 1000,
            "tune": 1000,
            "cores": 4,
            "chains": 4,
            "target_accept": 0.98,
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """

        pass

    def _generate_and_preprocess_model_data(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y




class AreaCountInteraction1DPartPoolPositive(ModelBuilder):
    # Give the model a name
    _model_type = "AreaCountInteraction1DPartPoolPositive"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model for making predictions on new areas

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """

        area_idx_int, self.area_map = pd.factorize(X['area_id'], sort=True)
        self.n_areas = len(self.area_map)

        coords = {
            "area": self.area_map,
        }

        # Check the type of X and y and adjust access accordingly
        self._generate_and_preprocess_model_data(X, y)

        y = y.values if isinstance(y, pd.Series) else y

        with pm.Model(coords=coords) as self.model:
            # Create data containers
            area_idx = pm.Data("area_idx", area_idx_int)

            x_data = pm.Data(
                    "x_data", 
                    X['total_counts'].values,
                )
            m_data = pm.Data("m_data", X['area_size'].values)
            y_obs = pm.Data("y_obs", y)

            β1_mu_prior         = self.model_config.get("β1_mu_prior", 0.01)
            β1_sigma_prior      = self.model_config.get("β1_sigma_prior", 2.0)
            β2_mean_mu_prior    = self.model_config.get("β2_mean_mu_prior", 0.1)
            β2_mean_sigma_prior = self.model_config.get("β2_mean_sigma_prior", 20.0)
            β2_std_lambda_prior = self.model_config.get("β2_std_lambda_prior", 5.0)

            σ_prior = self.model_config.get("σ_prior", 1000.0)

            # global coefficient priors
            β1 = pm.Gamma("β1", mu=β1_mu_prior, sigma=β1_sigma_prior)

            # group-specific area-count interaction term priors
            mu_b2 = pm.Gamma("mu_b", mu=β2_mean_mu_prior, sigma=β2_mean_sigma_prior)
            sigma_b2 = pm.Exponential("sigma_b", β2_std_lambda_prior)
            β2 = pm.Gamma("β2", mu=mu_b2, sigma=sigma_b2, dims="area")

            # model error
            σ = pm.HalfCauchy("σ", σ_prior)

            # linear mean 
            mu = β1*x_data + β2[area_idx]*x_data*m_data

            # likelihood
            obs = pm.Gamma(
                "y", 
                mu=mu, 
                sigma=σ, 
                observed=y_obs, 
            )

    def _data_setter(
        self, 
        X: pd.DataFrame, 
        y: pd.Series= None
    ):
        area_idx_int = self.area_map.get_indexer(X['area_id'])
        
        with self.model:
            pm.set_data({
                "x_data": X.total_counts.values,
                "m_data": X.area_size.values,
                "area_idx": area_idx_int,
            })
            if y is not None:
                pm.set_data({"y_obs": y.values})

    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config: Dict = { 
            "β1_mu_prior": 0.01,
            "β1_sigma_prior": 2.0,
            "β2_mean_mu_prior": 0.1,
            "β2_mean_sigma_prior": 20.0,
            "β2_std_lambda_prior": 5.0,
            "σ_prior": 1000.0
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: Dict = {
            "draws": 1000,
            "tune": 1000,
            "cores": 4,
            "chains": 4,
            "target_accept": 0.98,
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """

        pass

    def _generate_and_preprocess_model_data(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y



