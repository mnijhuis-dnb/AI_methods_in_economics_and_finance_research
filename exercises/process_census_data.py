from pathlib import Path
import bz2
import shutil

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "../data/census-income.data"  

DATA_TYPES = [
    ("age", "float64"),
    ("class of worker", "category"),
    ("detailed industry recode", "category"),
    ("detailed occupation recode", "category"),
    ("education", "category"),
    ("wage per hour", "float64"),
    ("enroll in edu inst last wk", "category"),
    ("marital stat", "category"),
    ("major industry code", "category"),
    ("major occupation code", "category"),
    ("race", "category"),
    ("hispanic origin", "category"),
    ("sex", "category"),
    ("member of a labor union", "category"),
    ("reason for unemployment", "category"),
    ("full or part time employment stat", "category"),
    ("capital gains", "float64"),
    ("capital losses", "float64"),
    ("dividends from stocks", "float64"),
    ("tax filer stat", "category"),
    ("region of previous residence", "category"),
    ("state of previous residence", "category"),
    ("detailed household and family stat", "category"),
    ("detailed household summary in household", "category"),
    ("instance weight_ignore", "float64"),
    ("migration code-change in msa", "category"),
    ("migration code-change in reg", "category"),
    ("migration code-move within reg", "category"),
    ("live in this house 1 year ago", "category"),
    ("migration prev res in sunbelt", "category"),
    ("num persons worked for employer", "float64"),
    ("family members under 18", "category"),
    ("country of birth father", "category"),
    ("country of birth mother", "category"),
    ("country of birth self", "category"),
    ("citizenship", "category"),
    ("own business or self employed", "category"),
    ("fill inc questionnaire for veteran's admin", "category"),
    ("veterans benefits", "category"),
    ("weeks worked in year", "float64"),
    ("year", "category"),
    ("targets", "category"),
]

EDU_CODE = {
    "Children": 0,
    "Less than 1st grade": 1,
    "1st 2nd 3rd or 4th grade": 2,
    "5th or 6th grade": 3,
    "7th and 8th grade": 4,
    "9th grade": 5,
    "10th grade": 6,
    "11th grade": 7,
    "12th grade no diploma": 8,
    "High school graduate": 9,
    "Some college but no degree": 10,
    "Associates degree-academic program": 11,
    "Associates degree-occup /vocational": 12,
    "Bachelors degree(BA AB BS)": 13,
    "Masters degree(MA MS MEng MEd MSW MBA)": 14,
    "Prof school degree (MD DDS DVM LLB JD)": 15,
    "Doctorate degree(PhD EdD)": 15,
}

BINARY_COLUMNS = [
    "member of a labor union",
    "live in this house 1 year ago",
    "own business or self employed",
    "fill inc questionnaire for veteran's admin",
    "veterans benefits",
    "year",
    "sex",
]

def load_raw_data(path):
    """Load the raw census income dataset into a pandas DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path or None
        Path to the uncompressed dataset file. When ``None``, the default
        module path is used and the compressed file is extracted if needed.

    Returns
    -------
    pandas.DataFrame
        Raw census data with column names and dtypes defined by ``DATA_TYPES``.
    """
    if path is None:
        path = DATA_PATH
        if not Path(path).exists():
            _unzip_data()

    return pd.read_csv(path, names=[d[0] for d in DATA_TYPES], dtype=dict(DATA_TYPES))

def _unzip_data():
    """Extract the bundled compressed census income dataset.

    Returns
    -------
    None
        The function writes the extracted file next to the compressed archive.
    """

    source_path = Path(DATA_PATH + ".bz2")
    target_path = source_path.with_suffix("")

    with bz2.open(source_path, "rb") as src, target_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def add_education_num(raw_data):
    """Add a numeric encoding of the education label column.

    Parameters
    ----------
    raw_data : pandas.DataFrame
        Input census dataset containing the ``education`` column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``raw_data`` with an additional ``education-num`` column.
    """
    raw_data = raw_data.copy()
    raw_data["education-num"] = np.array(
        [EDU_CODE[v.strip()] for v in raw_data["education"]]
    ).astype("float64")
    return raw_data


def build_features(raw_data):
    """Transform raw census records into a model-ready feature matrix.

    Parameters
    ----------
    raw_data : pandas.DataFrame
        Census records without the target column.

    Returns
    -------
    pandas.DataFrame
        Feature matrix containing one-hot encoded categorical features,
        numerical columns, and factorized binary columns.
    """
    raw_data = add_education_num(raw_data)
    data = raw_data.drop(
        columns=[
            "instance weight_ignore",
            "detailed industry recode",
            "detailed occupation recode",
            "major industry code",
            "major occupation code",
            "country of birth father",
            "country of birth mother",
            "country of birth self",
            "state of previous residence",
            "detailed household and family stat",
            "education",
        ]
    )

    binary_data = data[BINARY_COLUMNS].copy()
    categorical_data = data.select_dtypes(include=["category"]).drop(
        columns=BINARY_COLUMNS, errors="ignore"
    )
    numerical_data = data.select_dtypes(include=["float64", "int64"]).drop(
        columns=BINARY_COLUMNS, errors="ignore"
    )

    binary_data[
        (binary_data == " 2")
        | (binary_data == " ?")
        | (binary_data == " Not in universe")
        | (binary_data == " Not in universe under 1 year old")
    ] = np.nan
    binary_data = binary_data.apply(lambda x: pd.factorize(x)[0])

    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(categorical_data)

    categorical_source = data.select_dtypes(include=["category"]).drop(
        columns=BINARY_COLUMNS, errors="ignore"
    )

    return pd.concat(
        [
            pd.DataFrame(
                encoder.transform(categorical_source).toarray(),
                index=categorical_source.index,
                columns=encoder.get_feature_names_out(categorical_source.columns),
            ),
            numerical_data,
            binary_data,
        ],
        axis=1,
    )


def rebalance_training_data(train_x, train_y, random_state=99):
    """Downsample the majority class to create a balanced training set.

    Parameters
    ----------
    train_x : pandas.DataFrame
        Training features.
    train_y : pandas.Series
        Binary target labels aligned with ``train_x``.
    random_state : int, default=99
        Seed used for deterministic resampling.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Resampled training features and labels with balanced classes.
    """
    rng = np.random.default_rng(random_state)
    positive_idx = train_y.index.values[train_y == 1]
    negative_idx = rng.permutation(train_y.index.values[train_y == 0])
    index = rng.permutation(np.hstack((positive_idx, negative_idx))[: 2 * len(positive_idx)])
    return train_x.loc[index], train_y.loc[index]


def prepare_census_data(
    path=None,
    random_state=99,
    split_validation=False,
    rebalance=True,
):
    """Load, featurize, split, and optionally rebalance the census dataset.

    Parameters
    ----------
    path : str or pathlib.Path or None, default=None
        Path to the raw census data file. When ``None``, the default dataset
        path is used.
    random_state : int, default=99
        Seed used for train-test splitting and optional rebalancing.
    split_validation : bool, default=False
        Whether to create a separate validation split in addition to the test
        split.
    rebalance : bool, default=True
        Whether to rebalance the training data after splitting.

    Returns
    -------
    tuple
        Either ``(train_x, test_x, train_y, test_y)`` or
        ``(train_x, val_x, test_x, train_y, val_y, test_y)`` depending on
        ``split_validation``.
    """
    raw_data = load_raw_data(path)
    targets, _ = pd.factorize(raw_data["targets"])
    features = build_features(raw_data.drop(columns=["targets"]))
    target_series = pd.Series(targets, index=features.index)

    if split_validation:
        train_x, temp_x, train_y, temp_y = train_test_split(
            features, target_series, random_state=random_state
        )
        val_x, test_x, val_y, test_y = train_test_split(
            temp_x, temp_y, random_state=random_state
        )
        if rebalance:
            train_x, train_y = rebalance_training_data(train_x, train_y, random_state)
        return train_x, val_x, test_x, train_y, val_y, test_y

    train_x, test_x, train_y, test_y = train_test_split(
        features, target_series, random_state=random_state
    )
    if rebalance:
        train_x, train_y = rebalance_training_data(train_x, train_y, random_state)
    return train_x, test_x, train_y, test_y
