from src.data import load_data
from src.preprocessing import (
    CATEGORICAL_COLS,
    FEATURE_NAMES,
    NUMERIC_COLS,
    build_preprocessor,
    get_feature_defaults,
    prepare_features,
)


N_NUMERIC = 48
N_CATEGORICAL = 35
N_FEATURES = N_NUMERIC + N_CATEGORICAL  # 83


def test_prepare_features_train_returns_y():
    treino, _ = load_data()
    X, y = prepare_features(treino, has_target=True)
    assert X.shape[1] == N_FEATURES
    assert y is not None
    assert y.min() > 0
    assert "SalePrice" not in X.columns
    assert "Id" not in X.columns


def test_prepare_features_test_returns_none_y():
    _, teste = load_data()
    X, y = prepare_features(teste, has_target=False)
    assert X.shape[1] == N_FEATURES
    assert y is None


def test_get_feature_defaults_keys_and_types():
    defaults = get_feature_defaults()
    assert len(defaults) == N_FEATURES
    assert set(defaults.keys()) == set(FEATURE_NAMES)
    for c in NUMERIC_COLS:
        assert isinstance(defaults[c], float), f"{c} deveria ser float"
    for c in CATEGORICAL_COLS:
        assert isinstance(defaults[c], str), f"{c} deveria ser str"


def test_get_feature_defaults_returns_copy():
    a = get_feature_defaults()
    a["LotArea"] = 9_999_999.0
    b = get_feature_defaults()
    assert b["LotArea"] != 9_999_999.0


def test_feature_names_no_duplicates():
    assert len(FEATURE_NAMES) == N_FEATURES
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert len(NUMERIC_COLS) == N_NUMERIC
    assert len(CATEGORICAL_COLS) == N_CATEGORICAL


def test_preprocessor_expands_via_onehot():
    treino, _ = load_data()
    X, _ = prepare_features(treino)
    prep = build_preprocessor()
    Xt = prep.fit_transform(X)
    # 35 cat cols expandem; total deve crescer (cada cat com k níveis vira k colunas)
    assert Xt.shape[1] > N_FEATURES
    assert Xt.shape[0] == X.shape[0]
