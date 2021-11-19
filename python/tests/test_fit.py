import pytest

import l0learn


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_X_checks(f):
    with pytest.raises(ValueError):
        f()