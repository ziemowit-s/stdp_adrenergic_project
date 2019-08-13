import argparse

from pyvis.network import Network
import networkx as nx
import xml.etree.ElementTree as ET


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


def create_graph(reactions, reactants_left=None,
                 height="100%", width="100%", bgcolor="#222222", font_color="white"):
    def is_add(name):
        if reactants_left is None:
            return True

        for r in reactants_left:
            if r in name:
                return True

        return False

    g = Network(height=height, width=width, bgcolor=bgcolor,
                font_color=font_color, directed=True)

    nodes = []
    for k, v in reactions.items():
        for r in v['reactant']:
            for p in v['product']:

                if not is_add(r) and not is_add(p):
                    continue

                if r not in nodes:
                    nodes.append(r)
                    g.add_node(r)
                if p not in nodes:
                    nodes.append(p)
                    g.add_node(p)
                if k not in nodes:
                    nodes.append(k)
                    g.add_node(n_id=k, shape="dot", color="gray", label="R", size=10)

                fr = float(v['forward'])
                rr = float(v['reverse'])

                if fr > rr:
                    g.add_edge(k, p, arrowStrikethrough=True, value=fr, title="FOR:%s" % v['forward'])
                    g.add_edge(r, k, arrowStrikethrough=True, value=rr, title="REV:%s" % v['reverse'])
                else:
                    g.add_edge(p, k, arrowStrikethrough=True, value=fr, title="FOR:%s" % v['forward'])
                    g.add_edge(k, r, arrowStrikethrough=True, value=rr, title="REV:%s" % v['reverse'])

    return g


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaction_file")
    ap.add_argument("--reactants", nargs='+', help="If you want to only show some reactants", default=None)
    ap.add_argument("--removep", action='store_true', help="If you want to reduce number of particles on the graph, "
                                                           "by merging all phospho-particles (p..) to same particle.")
    args = ap.parse_args()

    _, reactions = parse_rx(filename=args.reaction_file, remove_p=args.removep)
    graph = create_graph(reactions=reactions, reactants_left=args.reactants)

    graph.show_buttons(filter_=['physics'])
    graph.hrepulsion(node_distance=185)
    graph.show('rx_graph.html')
