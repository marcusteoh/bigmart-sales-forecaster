"""Preprocessing for the BigMart sales model.

Imported by the notebook and the Streamlit app so a row is transformed identically in
training and in production. Fitted steps are transformers, so they only ever see the
training split.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Data collected in 2013. Age is measured against that year so the app never
# extrapolates past the oldest outlet the model saw.
DATA_YEAR = 2013

# Five spellings, two real categories (notebook 2.2.4).
FAT_CONTENT_MAP = {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}

# These three Item_Types are exactly the 'NC' non-consumable prefix (notebook 3.1.1).
# Keying off Item_Type lets the app apply the rule without a product code.
NON_EDIBLE_ITEM_TYPES = ('Health and Hygiene', 'Household', 'Others')
NON_EDIBLE_LABEL = 'Non-Edible'

# Only Item_MRP, Outlet_Type and Outlet_Age carry real signal. All nine are kept
# because notebook 3.5 measured the alternatives: CV RMSE spread 3.90 against fold
# noise of 20 to 23, so removal is a change inside the noise. Identifiers are
# excluded (notebook 2.2.6).
NUMERIC_FEATURES = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
CATEGORICAL_FEATURES = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size',
                        'Outlet_Location_Type', 'Outlet_Type']


def clean_deterministic(X):
    """Fixed rules that need no statistic from the data, so they cannot leak."""
    X = X.copy()

    # Collapse the five fat-content spellings into two.
    X['Item_Fat_Content'] = X['Item_Fat_Content'].replace(FAT_CONTENT_MAP)

    # Fat content is meaningless for cleaning products, so give them their own label.
    is_non_edible = X['Item_Type'].isin(NON_EDIBLE_ITEM_TYPES)
    X.loc[is_non_edible, 'Item_Fat_Content'] = NON_EDIBLE_LABEL

    # A stocked item cannot occupy 0% of the shelf, so treat zero as missing.
    X['Item_Visibility'] = X['Item_Visibility'].replace(0.0, np.nan)

    # Three outlets have no size recorded. Say so rather than invent one.
    X['Outlet_Size'] = X['Outlet_Size'].fillna('Unknown')

    # Years trading is the business quantity, and it is monotonic unlike a raw year.
    X['Outlet_Age'] = DATA_YEAR - X['Outlet_Establishment_Year']

    return X


class ItemWeightImputer(BaseEstimator, TransformerMixin):
    """Fill a missing Item_Weight from the same product's weight on another row.

    Weight is a stable product attribute, so this recovers 1,138 of the 1,142 affected
    products where a global median would not. The lookup is built in fit(), so it is
    learned from the training split only, and it is skipped when Item_Identifier is
    absent as it is in the app.
    """

    def fit(self, X, y=None):
        if 'Item_Identifier' in X.columns:
            self.weight_by_item_ = (
                X.groupby('Item_Identifier')['Item_Weight'].median().dropna()
            )
        else:
            self.weight_by_item_ = pd.Series(dtype=float)
        self.global_median_ = X['Item_Weight'].median()
        return self

    def transform(self, X):
        X = X.copy()
        if 'Item_Identifier' in X.columns and len(self.weight_by_item_):
            recovered = X['Item_Identifier'].map(self.weight_by_item_)
            X['Item_Weight'] = X['Item_Weight'].fillna(recovered)
        X['Item_Weight'] = X['Item_Weight'].fillna(self.global_median_)
        return X


def build_preprocessor(numeric_features, categorical_features):
    """Impute and scale the numerics, impute and one-hot the categoricals.

    Every step is fitted, so it must sit inside the pipeline. Unnamed columns are
    dropped. StandardScaler is needed by Linear Regression and harmless to the trees.
    """
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' stops the app crashing on an unseen category.
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ('num', numeric_pipe, list(numeric_features)),
            ('cat', categorical_pipe, list(categorical_features)),
        ],
        remainder='drop',
    )


def build_pipeline(model, numeric_features=NUMERIC_FEATURES,
                   categorical_features=CATEGORICAL_FEATURES):
    """The complete path from a raw BigMart row to a prediction."""
    return Pipeline([
        ('clean', FunctionTransformer(clean_deterministic, validate=False)),
        ('weight', ItemWeightImputer()),
        ('prep', build_preprocessor(numeric_features, categorical_features)),
        ('model', model),
    ])
