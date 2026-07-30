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

# the BigMart data was collected in 2013. outlet age is measured against that year so
# the feature keeps the same meaning when the app runs in a later calendar year - if
# the app used the real current year, every outlet would look older than anything the
# model was trained on, and the model would be extrapolating.
DATA_YEAR = 2013

# the raw column holds five spellings for what are only two real categories (notebook
# 2.2.4). left as they are, the encoder would produce five columns and split one signal
# across duplicates that the model has no way of knowing are the same thing.
FAT_CONTENT_MAP = {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}

# notebook 3.1.1 verified that these three Item_Types are exactly the products carrying
# the 'NC' (non-consumable) identifier prefix, and nothing else. the rule keys off
# Item_Type rather than Item_Identifier so the Streamlit app can apply the identical
# rule without asking a user for a product code they have no reason to know.
NON_EDIBLE_ITEM_TYPES = ('Health and Hygiene', 'Household', 'Others')
NON_EDIBLE_LABEL = 'Non-Edible'

# the full candidate feature set. Outlet_Age replaces the raw establishment year, and
# the identifier columns are deliberately absent (see notebook 2.2.6).
#
# WHY ALL NINE ARE KEPT, INCLUDING THE ONES THAT BARELY MOVE THE PREDICTION
# ------------------------------------------------------------------------
# two independent measurements agree that only three features carry real signal:
#
#   feature                     eta-squared   permutation importance
#   Item_MRP                        0.319       707.15  (+/- 34.55)
#   Outlet_Type                     0.240       545.75  (+/- 23.85)
#   Outlet_Establishment_Year       0.087        28.61  (+/-  6.83)
#   Outlet_Size                     0.048         0.13  (+/-  0.43)
#   Item_Visibility                 0.017        -0.19  (+/-  0.68)
#   Outlet_Location_Type            0.013         0.01  (+/-  0.09)
#   Item_Type                       0.005         0.17  (+/-  0.65)
#   Item_Weight                     0.002        -0.04  (+/-  0.86)
#   Item_Fat_Content               0.0004        -0.28  (+/-  0.19)
#
# the bottom six are indistinguishable from zero. Outlet_Location_Type is the most
# extreme: sweeping 720 input combinations moves the forecast by exactly 0.00, because
# city tier is confounded with store format across the ten outlets - Tier 2 contains
# only Supermarket Type1 stores, and Types 2 and 3 exist only in Tier 3, so once the
# format is known the tier carries no information the model does not already have.
#
# they are kept anyway, on evidence rather than habit. notebook 3.5 compared three
# feature sets by 5-fold CV on the training split:
#
#   Full    (9 features)   CV RMSE 1,100.4  +/- 23.1
#   Lean    (6 features)   CV RMSE 1,096.5  +/- 21.7
#   Minimal (3 features)   CV RMSE 1,099.3  +/- 20.1
#
# the spread across all three is 3.90 while the fold-to-fold standard deviation is
# 20-23, five times larger. removing the weak features is therefore not an improvement,
# it is a change inside the noise. with no measurable difference to decide it, the tie
# is broken on deployment grounds - the Streamlit app is more useful to a category
# manager who can vary product type, weight and fat content, and the nine features are
# the set the approved project proposal specified.
#
# the finding itself is the real result. this problem is limited by the information in
# the dataset - no footfall, promotions, competitor pricing or stock-on-hand - not by
# the model or the feature list. if this went to production at scale the Minimal set
# would be the right choice, three inputs instead of nine at no measurable cost in
# accuracy.
NUMERIC_FEATURES = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
CATEGORICAL_FEATURES = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size',
                        'Outlet_Location_Type', 'Outlet_Type']


def clean_deterministic(X):
    """Fixed rules that need no statistic from the data, so they cannot leak.

    Safe to apply before the train/test split, but kept inside the pipeline anyway so
    that a raw row from the app travels the exact same path as a training row.
    """
    X = X.copy()

    # collapse the five fat-content spellings into the two categories they actually
    # represent, so the encoder sees one column per category rather than five.
    X['Item_Fat_Content'] = X['Item_Fat_Content'].replace(FAT_CONTENT_MAP)

    # fat content is meaningless for cleaning products and toiletries. giving them their
    # own label keeps 1,599 rows from masquerading as 'Low Fat' food, which would water
    # down whatever that category means for the products where it does mean something.
    is_non_edible = X['Item_Type'].isin(NON_EDIBLE_ITEM_TYPES)
    X.loc[is_non_edible, 'Item_Fat_Content'] = NON_EDIBLE_LABEL

    # a stocked item that recorded sales cannot occupy 0% of the shelf, so the zero is a
    # missing value in disguise. turning it into a real NaN hands it to the fitted
    # imputer - left as 0.0 the model would read it as a genuine measurement of the
    # least visible product in the chain.
    X['Item_Visibility'] = X['Item_Visibility'].replace(0.0, np.nan)

    # three outlets have no size on record at all (notebook 2.2.7). 'Unknown' says that
    # plainly, where imputing the mode would invent a measurement for 28% of rows and
    # hide a real fact about those stores from the model and from anyone reading it.
    X['Outlet_Size'] = X['Outlet_Size'].fillna('Unknown')

    # years trading is the quantity a planner actually reasons about, and unlike a raw
    # calendar year it is monotonic - a larger value always means an older outlet, so
    # the model can use it as a magnitude instead of an arbitrary label.
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
        # handle_unknown='ignore' encodes an unseen category as all zeros instead of
        # raising. the app takes free-form input from a user, so without this a single
        # category the training split never contained would take the whole page down.
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
