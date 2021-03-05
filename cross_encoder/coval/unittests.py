from pytest import approx
from coval.conll.reader import get_coref_infos
from coval.eval.evaluator import evaluate_documents as evaluate
from coval.eval.evaluator import muc, b_cubed, ceafe, lea

TOL = 1e-4


def read(key, response):
    return get_coref_infos('tests/%s' % key, 'tests/%s' % response,
            False, False, True)


def test_A1():
    doc = read('TC-A.key', 'TC-A-1.response')
    assert evaluate(doc, muc) == (1, 1, 1)
    assert evaluate(doc, b_cubed) == (1, 1, 1)
    assert evaluate(doc, ceafe) == (1, 1, 1)
    assert evaluate(doc, lea) == (1, 1, 1)


def test_A2():
    doc = read('TC-A.key', 'TC-A-2.response')
    assert evaluate(doc, muc) == approx([1 / 3, 1 / 1, 1 / 2])
    assert evaluate(doc, b_cubed) == approx([(7 / 3) / 6, 3 / 3, 14 / 25])
    assert evaluate(doc, ceafe) == approx([0.6, 0.9, 0.72])
    assert evaluate(doc, lea) == approx([(1 + 3 * (1 / 3)) / 6, 1, 0.5])


def test_A3():
    doc = read('TC-A.key', 'TC-A-3.response')
    assert evaluate(doc, muc) == approx([3 / 3, 3 / 5, 0.75])
    assert evaluate(doc,
            b_cubed) == approx([6 / 6, (4 + 7 / 12) / 9, 110 / 163])
    assert evaluate(doc, ceafe) == approx([0.88571, 0.66429, 0.75918], abs=TOL)
    assert evaluate(doc, lea) == approx([
            1, (1 + 3 * (1 / 3) + 4 * (3 / 6)) / 9,
            2 * (1 + 3 * (1 / 3) + 4
                * (3 / 6)) / 9 / (1 + (1 + 3 * (1 / 3) + 4 * (3 / 6)) / 9)
    ])


def test_A4():
    doc = read('TC-A.key', 'TC-A-4.response')
    assert evaluate(doc, muc) == approx([1 / 3, 1 / 3, 1 / 3])
    assert evaluate(doc, b_cubed) == approx([
            (3 + 1 / 3) / 6, (1 + 4 / 3 + 1 / 2) / 7,
            2 * (5 / 9) * (17 / 42) / ((5 / 9) + (17 / 42))
    ])
    assert evaluate(doc, ceafe) == approx([0.73333, 0.55, 0.62857], abs=TOL)
    assert evaluate(doc, lea) == approx([(1 + 2 + 0) / 6,
            (1 + 3 * (1 / 3) + 2 * 0 + 0) / 7,
            2 * 0.5 * 2 / 7 / (0.5 + 2 / 7)])


def test_A5():
    doc = read('TC-A.key', 'TC-A-5.response')
    assert evaluate(doc, muc) == approx([1 / 3, 1 / 4, 2 / 7])
    assert evaluate(doc, b_cubed) == approx([
            (3 + 1 / 3) / 6, 2.5 / 8,
            2 * (5 / 9) * (5 / 16) / ((5 / 9) + (5 / 16))
    ])
    assert evaluate(doc, ceafe) == approx([0.68889, 0.51667, 0.59048], abs=TOL)
    assert evaluate(doc,
            lea) == approx([(1 + 2 + 3 * 0) / 6,
            (1 + 4 * (1 / 6) + 2 * 0 + 1 * 0) / 8,
            2 * 0.5 * (5 / 24) / (0.5 + (5 / 24))])


def test_A6():
    doc = read('TC-A.key', 'TC-A-6.response')
    assert evaluate(doc, muc) == approx([1 / 3, 1 / 4, 2 / 7])
    assert evaluate(doc, b_cubed) == approx([
            (10 / 3) / 6, (1 + 4 / 3 + 1 / 2) / 8,
            2 * (5 / 9) * (17 / 48) / ((5 / 9) + (17 / 48))
    ])
    assert evaluate(doc, ceafe) == approx([0.73333, 0.55, 0.62857], abs=TOL)
    assert evaluate(doc, lea) == approx([(1 + 2 + 3 * 0) / 6,
            (1 + 3 / 3 + 2 * 0 + 2 * 0) / 8,
            2 * 0.5 * 1 / 4 / (0.5 + 1 / 4)])


def test_A7():
    doc = read('TC-A.key', 'TC-A-7.response')
    assert evaluate(doc, muc) == approx([1 / 3, 1 / 3, 1 / 3])
    assert evaluate(doc, b_cubed) == approx([
            (10 / 3) / 6, (1 + 4 / 3 + 1 / 2) / 7,
            2 * (5 / 9) * (17 / 42) / ((5 / 9) + (17 / 42))
    ])
    assert evaluate(doc, ceafe) == approx([0.73333, 0.55, 0.62857], abs=TOL)
    assert evaluate(doc, lea) == approx([(1 + 2 + 3 * 0) / 6,
            (1 + 3 / 3 + 2 * 0 + 1 * 0) / 7,
            2 * 0.5 * 2 / 7 / (0.5 + 2 / 7)])


def test_A10():
    doc = read('TC-A.key', 'TC-A-10.response')
    assert evaluate(doc, muc) == approx([0, 0, 0])
    assert evaluate(doc, b_cubed) == approx([3 / 6, 6 / 6, 2 / 3])
    assert evaluate(doc, lea) == approx(
            [1 / 6, 1 / 6, 2 * 1 / 6 * 1 / 6 / (1 / 6 + 1 / 6)])


def test_A11():
    doc = read('TC-A.key', 'TC-A-11.response')
    assert evaluate(doc, muc) == approx([3 / 3, 3 / 5, 6 / 8])
    assert evaluate(doc, b_cubed) == approx(
            [6 / 6, (1 / 6 + 2 * 2 / 6 + 3 * 3 / 6) / 6, 14 / 25])
    assert evaluate(doc,
            lea) == approx([(0 + 2 + 3) / 6, 4 / 15,
            2 * 5 / 6 * 4 / 15 / (5 / 6 + 4 / 15)])


def test_A12():
    doc = read('TC-A.key', 'TC-A-12.response')
    assert evaluate(doc, muc) == approx([0, 0, 0])
    assert evaluate(doc, b_cubed) == approx([
            (1 + 1 / 2 + 2 / 3) / 6, 4 / 7,
            2 * (13 / 36) * (4 / 7) / ((13 / 36) + (4 / 7))
    ])
    assert evaluate(doc, lea) == approx(
            [1 / 6, 1 / 7, 2 * 1 / 6 * 1 / 7 / (1 / 6 + 1 / 7)])


def test_A13():
    doc = read('TC-A.key', 'TC-A-13.response')
    assert evaluate(doc, muc) == approx([1 / 3, 1 / 6, 2 / 9])
    assert evaluate(doc, b_cubed) == approx([
            (1 + 1 / 2 + 2 * 2 / 3) / 6, (1 / 7 + 1 / 7 + 2 * 2 / 7) / 7,
            2 * (17 / 36) * (6 / 49) / ((17 / 36) + (6 / 49))
    ])
    assert evaluate(doc,
            lea) == approx([(1 * 0 + 2 * 0 + 3 / 3) / 6, 1 / 21,
            2 * 1 / 6 * 1 / 21 / (1 / 6 + 1 / 21)])


def test_B1():
    doc = read('TC-B.key', 'TC-B-1.response')
    assert evaluate(doc, lea) == approx([(2 * 0 + 3 / 3) / 5, (3 * 0 + 2) / 5,
            2 * 1 / 5 * 2 / 5 / (1 / 5 + 2 / 5)])


def test_C1():
    doc = read('TC-C.key', 'TC-C-1.response')
    assert evaluate(doc, lea) == approx([(2 * 0 + 3 / 3 + 2) / 7,
            (3 * 0 + 2 + 2) / 7,
            2 * 3 / 7 * 4 / 7 / (3 / 7 + 4 / 7)])


def test_D1():
    doc = read('TC-D.key', 'TC-D-1.response')
    assert evaluate(doc, muc) == approx(
            [9 / 9, 9 / 10, 2 * (9 / 9) * (9 / 10) / (9 / 9 + 9 / 10)])
    assert evaluate(doc, b_cubed) == approx([
            12 / 12, 16 / 21, 2 * (12 / 12) * (16 / 21) / (12 / 12 + 16 / 21)
    ])
    assert evaluate(doc, lea) == approx([
            (5 + 2 + 5) / 12, (5 + 7 * (11 / 21)) / 12,
            2 * 1 * (5 + 77 / 21) / 12 / (1 + ((5 + 77 / 21) / 12))
    ])


def test_E1():
    doc = read('TC-E.key', 'TC-E-1.response')
    assert evaluate(doc, muc) == approx(
            [9 / 9, 9 / 10, 2 * (9 / 9) * (9 / 10) / (9 / 9 + 9 / 10)])
    assert evaluate(doc, b_cubed) == approx(
            [1, 7 / 12, 2 * 1 * (7 / 12) / (1 + 7 / 12)])
    assert evaluate(doc, lea) == approx([(5 + 2 + 5) / 12,
            (10 * (20 / 45) + 2) / 12,
            2 * 1 * ((10 * (20 / 45) + 2) / 12)
                / (1 + ((10 * (20 / 45) + 2) / 12))])


def test_F1():
    doc = read('TC-F.key', 'TC-F-1.response')
    assert evaluate(doc, muc) == approx(
            [2 / 3, 2 / 2, 2 * (2 / 3) * (2 / 2) / (2 / 3 + 2 / 2)])
    assert evaluate(doc, lea) == approx(
            [4 * (2 / 6) / 4, (2 + 2) / 4, 2 * 2 / 6 * 1 / (1 + 2 / 6)])


def test_G1():
    doc = read('TC-G.key', 'TC-G-1.response')
    assert evaluate(doc, muc) == approx(
            [2 / 2, 2 / 3, 2 * (2 / 2) * (2 / 3) / (2 / 2 + 2 / 3)])
    assert evaluate(doc, lea) == approx(
            [1, (4 * 2 / 6) / 4, 2 * 1 * 2 / 6 / (1 + 2 / 6)])


def test_H1():
    doc = read('TC-H.key', 'TC-H-1.response')
    assert evaluate(doc, muc) == approx([1, 1, 1])
    assert evaluate(doc, lea) == approx([1, 1, 1])


def test_I1():
    doc = read('TC-I.key', 'TC-I-1.response')
    assert evaluate(doc, muc) == approx(
            [2 / 3, 2 / 2, 2 * (2 / 3) * (2 / 2) / (2 / 3 + 2 / 2)])
    assert evaluate(doc, lea) == approx(
            [4 * (2 / 6) / 4, (2 + 2) / 4, 2 * 2 / 6 * 1 / (2 / 6 + 1)])


def test_J1():
    doc = read('TC-J.key', 'TC-J-1.response')
    assert evaluate(doc, muc) == approx(
            [1 / 2, 1 / 1, 2 * (1 / 2) * (1 / 1) / (1 / 2 + 1 / 1)])
    assert evaluate(doc, lea) == approx([(3 * 1 / 3) / 3, 1,
            2 * 1 / 3 / (1 + 1 / 3)])


def test_K1():
    doc = read('TC-K.key', 'TC-K-1.response')
    assert evaluate(doc, muc) == approx([3 / 6, 3 / 6, 3 / 6])
    assert evaluate(doc,
            lea) == approx([(7 * (1 + 1 + 1) / 21) / 7,
            (3 / 3 + 3 / 3 + 3 / 3) / 9,
            2 * 3 / 21 * 3 / 9 / (3 / 21 + 3 / 9)])


def test_L1():
    doc = read('TC-L.key', 'TC-L-1.response')
    assert evaluate(doc, muc) == approx(
            [2 / 5, 2 / 4, 2 * (2 / 5) * (2 / 4) / (2 / 5 + 2 / 4)])
    assert evaluate(doc, lea) == approx([
            (3 * 1 / 3 + 4 * 1 / 6) / 7, (2 + 2 * 0 + 3 / 3) / 7,
            2 * (1 + 2 / 3) / 7 * 3 / 7 / (3 / 7 + (1 + 2 / 3) / 7)
    ])


def test_M1():
    doc = read('TC-M.key', 'TC-M-1.response')
    assert evaluate(doc, muc) == approx([1, 1, 1])
    assert evaluate(doc, b_cubed) == approx([1, 1, 1])
    assert evaluate(doc, ceafe) == approx([1, 1, 1])
    assert evaluate(doc, lea) == approx([1, 1, 1])


def test_M2():
    doc = read('TC-M.key', 'TC-M-2.response')
    assert evaluate(doc, muc) == approx([0, 0, 0])
    assert evaluate(doc, lea) == approx([0, 0, 0])


def test_M3():
    doc = read('TC-M.key', 'TC-M-3.response')
    assert evaluate(doc, lea) == approx([
            6 * (4 / 15) / 6, (2 + 3 + 0) / 6,
            2 * 4 / 15 * 5 / 6 / (4 / 15 + 5 / 6)
    ])


def test_M4():
    doc = read('TC-M.key', 'TC-M-4.response')
    assert evaluate(doc, lea) == approx([
            6 * (3 / 15) / 6, 6 * (3 / 15) / 6,
            2 * 3 / 15 * 3 / 15 / (3 / 15 + 3 / 15)
    ])


def test_M5():
    doc = read('TC-M.key', 'TC-M-5.response')
    assert evaluate(doc, muc) == approx([0, 0, 0])
    assert evaluate(doc, lea) == approx([0, 0, 0])


def test_M6():
    doc = read('TC-M.key', 'TC-M-6.response')
    assert evaluate(doc, lea) == approx([
            6 * (1 / 15) / 6, (2 + 3 * 0 + 1 * 0) / 6,
            2 * 1 / 15 * 2 / 6 / (1 / 15 + 2 / 6)
    ])


def test_N1():
    doc = read('TC-N.key', 'TC-N-1.response')
    assert evaluate(doc, muc) == approx([0, 0, 0])
    assert evaluate(doc, lea) == approx([1, 1, 1])


def test_N2():
    doc = read('TC-N.key', 'TC-N-2.response')
    assert evaluate(doc, muc) == approx([0, 0, 0])
    assert evaluate(doc, lea) == approx([0, 0, 0])


def test_N3():
    doc = read('TC-N.key', 'TC-N-3.response')
    assert evaluate(doc, lea) == approx([1 / 6, 1 / 6, 1 / 6])


def test_N4():
    doc = read('TC-N.key', 'TC-N-4.response')
    assert evaluate(doc, muc) == approx([0, 0, 0])
    assert evaluate(doc, lea) == approx([3 / 6, 3 / 6, 3 / 6])


def test_N5():
    doc = read('TC-N.key', 'TC-N-5.response')
    assert evaluate(doc, lea) == approx([0, 0, 0])


def test_N6():
    doc = read('TC-N.key', 'TC-N-6.response')
    assert evaluate(doc, lea) == approx([0, 0, 0])
