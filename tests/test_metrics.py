from forseti.fairness import parity_score


def test_parity_score():
    probs = [0.5, 0.5]
    assert parity_score(probs) == 1
    probs = [1, 0]
    assert parity_score(probs) == 0
