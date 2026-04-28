from src.data import load_data
from src.preprocessing import (
    FEATURE_NAMES,
    get_feature_defaults,
    prepare_features,
)


def test_prepare_features_train_returns_y():
    treino, _ = load_data()
    X, y = prepare_features(treino, has_target=True)
    assert X.shape[1] == 48
    assert y is not None
    assert y.min() > 0
    assert "SalePrice" not in X.columns
    assert "Id" not in X.columns


def test_prepare_features_test_returns_none_y():
    _, teste = load_data()
    X, y = prepare_features(teste, has_target=False)
    assert X.shape[1] == 48
    assert y is None


def test_get_feature_defaults_keys_and_floats():
    defaults = get_feature_defaults()
    assert len(defaults) == 48
    assert set(defaults.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in defaults.values())


def test_get_feature_defaults_returns_copy():
    a = get_feature_defaults()
    a["LotArea"] = 9_999_999.0
    b = get_feature_defaults()
    assert b["LotArea"] != 9_999_999.0


def test_feature_names_no_duplicates():
    assert len(FEATURE_NAMES) == 48
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
