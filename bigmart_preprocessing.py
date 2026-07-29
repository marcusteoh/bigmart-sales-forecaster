"""Preprocessing for the BigMart sales model.

Imported by BOTH the notebook and the Streamlit app so that a row is transformed
identically in training and in production. Anything that needs a value learned from
data is a scikit-learn transformer, so it can be fitted on the training split only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# The BigMart data was collected in 2013. Outlet age is measured against that year so
# the feature keeps the same meaning when the app runs in a later calendar year - if
# the app used the real current year, every outlet would look older than anything the
# model was trained on, and the model would be extrapolating.
DATA_YEAR = 2013

# Five spellings, two real categories (notebook 2.2.4).
FAT_CONTENT_MAP = {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}

# Verified in notebook 3.1.1: these three Item_Types are exactly the 'NC' (non-consumable)
# identifier prefix, and nothing else. Keying off Item_Type instead of Item_Identifier
# lets the Streamlit app apply the identical rule without asking for a product code.
NON_EDIBLE_ITEM_TYPES = ('Health and Hygiene', 'Household', 'Others')
NON_EDIBLE_LABEL = 'Non-Edible'

# The full candidate feature set. Outlet_Age replaces the raw establishment year;
# the identifier columns are deliberately absent (see notebook 2.2.6).
NUMERIC_FEATURES = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
CATEGORICAL_FEATURES = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size',
                        'Outlet_Location_Type', 'Outlet_Type']


def clean_deterministic(X):
    """Fixed rules that need no statistic from the data, so they cannot leak.

    Safe to apply before the train/test split, but kept inside the pipeline anyway so
    that a raw row from the app travels the exact same path as a training row.
    """
    X = X.copy()

    # 1. Collapse the five fat-content spellings into the two real categories.
    X['Item_Fat_Content'] = X['Item_Fat_Content'].replace(FAT_CONTENT_MAP)

    # 2. Fat content is meaningless for cleaning products and toiletries. Give them
    #    their own label instead of letting 1,599 rows masquerade as 'Low Fat' food.
    is_non_edible = X['Item_Type'].isin(NON_EDIBLE_ITEM_TYPES)
    X.loc[is_non_edible, 'Item_Fat_Content'] = NON_EDIBLE_LABEL

    # 3. A stocked item that recorded sales cannot occupy 0%% of the shelf. Convert the
    #    disguised missing value into a real NaN so the fitted imputer handles it.
    X['Item_Visibility'] = X['Item_Visibility'].replace(0.0, np.nan)

    # 4. Three outlets have no size on record at all (notebook 2.2.7). Saying so is
    #    honest; imputing the mode would invent a measurement for 28%% of rows.
    X['Outlet_Size'] = X['Outlet_Size'].fillna('Unknown')

    # 5. Feature engineering: years trading is the business quantity a planner reasons
    #    about, and it is monotonic, unlike a raw calendar year.
    X['Outlet_Age'] = DATA_YEAR - X['Outlet_Establishment_Year']

    return X


class ItemWeightImputer(BaseEstimator, TransformerMixin):
    """Recover a missing Item_Weight from the same product measured at another outlet.

    Notebook 2.2.7 established that weight is a stable product attribute (at most one
    distinct weight per product) and that 1,138 of the 1,142 affected products have it
    recorded on another row. A per-product lookup therefore recovers almost all of them,
    where a global median would discard that information.

    The lookup table is built in fit(), so it is learned from the training split only.
    Falls back to the training median for products never seen (and works unchanged when
    Item_Identifier is absent, as it is in the Streamlit app).
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
    """Impute + scale the numerics, impute + one-hot the categoricals.

    Every step here is fitted, so it must sit inside the pipeline. Columns not named
    (the identifiers, the raw establishment year) are removed by remainder='drop'.
    StandardScaler is needed by Linear Regression and is harmless to the tree models,
    so one preprocessor can serve every candidate model.
    """
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' stops the app crashing if it is ever sent a category
        # the training split never contained.
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
