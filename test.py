def sorted(s, num):
    tmp = s.order(ascending=False)[:num]
    tmp.index = range(num)
    return tmp

def test_sorted()