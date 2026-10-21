import math
import random

from src.selfplay.elo import ANCHOR, fit, pairwise


def test_pairwise_order_ties_and_duplicates():
    # seat 2 wins on the tiebreak despite equal VP with seat 0; seats 1 and 3 are the same player
    res = pairwise(["a", "b", "c", "b"], [10, 7, 10, 7], winner=2)
    assert ("a", "c", 0.0) in res and ("a", "b", 1.0) in res and ("b", "c", 0.0) in res
    assert not any(x == y for x, y, _ in res)
    assert len(res) == 5


def test_fit_recovers_known_ratings():
    true = {ANCHOR: 1500, "weak": 1300, "strong": 1700, "top": 1850}
    names = list(true)
    r = random.Random(0)
    games = []
    for _ in range(6000):
        pl = r.sample(names, 4)
        # Plackett-Luce finishing order with strengths 10^(R/400) = consistent with Elo pairwise odds
        left, order = list(range(4)), []
        while left:
            w = [10 ** (true[pl[s]] / 400) for s in left]
            s = r.choices(left, w)[0]
            order.append(s); left.remove(s)
        vp = [0] * 4
        for k, s in enumerate(order):
            vp[s] = 10 - k
        games.append({"players": pl, "vp": vp, "winner": order[0]})
    rt = fit(games)
    assert rt[ANCHOR][0] == 1500
    for n in ("weak", "strong", "top"):
        assert abs(rt[n][0] - true[n]) < 3 * rt[n][1] + 15, (n, rt[n])
