import argparse

from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

LAYOUTS = {"kamada": nx.kamada_kawai_layout,
           "circular": nx.circular_layout,
           "bipartite": nx.bipartite_layout,
           "planar": nx.planar_layout,
           "shell": nx.shell_layout,
           "spring": nx.spring_layout,
           "spectral": nx.spectral_layout}


def parse_rx(filename, remove_p=False):
    doc = ET.parse(filename)

    species_kdiff = {}
    for s in doc.findall('Specie'):
        name = s.attrib['name']
        if remove_p:
            name = name.replace('p', '')
        species_kdiff[name] = float(s.attrib['kdiff'])

    reactions = {}
    for reaction in doc.findall("Reaction"):
        r = {'reactant': [], 'product': [], 'forward': 0, 'reverse': 0}
        reactions[reaction.attrib['name']] = r

        f = reaction.find("forwardRate")
        if f is not None:
            r['forward'] = float(f.text)

        b = reaction.find("reverseRate")
        if b is not None:
            r['reverse'] = float(b.text)

        for reactant in reaction.findall("Reactant"):
            name = reactant.attrib['specieID']
            if remove_p:
                name = name.replace('p', '')
            r['reactant'].append(name)

        for product in reaction.findall("Product"):
            name = product.attrib['specieID']
            if remove_p:
                name = name.replace('p', '')
            r['product'].append(name)

    return species_kdiff, reactions


def create_graph(species, reactions, reactants_left=None):
    def is_add(name):
        if reactants_left is None:
            return True

        for r in reactants_left:
            if r in name:
                return True

        return False

    g = nx.MultiDiGraph()
    nodes = []
    for s in species.keys():
        if is_add(s):
            nodes.append(s)

    for i, n in enumerate(nodes):
        g.add_node(n)

    for k, v in reactions.items():
        for r in v['reactant']:
            for p in v['product']:
                if is_add(r) or is_add(p):
                    g.add_edge(r, p)

    return g


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaction_file")
    ap.add_argument("--layout", help="allowed: kamada, circular, bipartite, planar, shell, spring, spectral",
                    default="shell")
    ap.add_argument("--reactants", nargs='+', help="If you want to only show some reactants", default=None)
    ap.add_argument("--removep", action='store_true', help="If you want to reduce number of particles on the graph, "
                                                           "by merging all phospho-particles (p..) to same particle.")
    args = ap.parse_args()

    species_kdiff, reactions = parse_rx(filename=args.reaction_file, remove_p=args.removep)
    graph = create_graph(species=species_kdiff, reactions=reactions, reactants_left=args.reactants)

    layout = LAYOUTS[args.layout](graph)
    nx.draw(graph, layout, with_labels=True, arrowsize=20)
    plt.show()
