from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt

a, b, c, d, e, f, g, h = range(8)

N = {
     a: [b, c, d, e, f],
     b: [c, e],
     c: [d],
     d: [e],
     e: [f],
     f: [c, g, h],
     g: [f, h],
     h: [f, g]
    }

G = nx.from_dict_of_lists(N)
nx.draw(G)
plt.show()
print(type(N))
pprint(N)

