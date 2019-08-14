import os
import argparse
from pyvis.network import Network
import xml.etree.ElementTree as ET


def neurord_parse_reaction_file(filename, remove_p=False):
    """

    :param filename:
    :param remove_p:
    :return:
    """
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
            r['reactant'].append(name)

        for product in reaction.findall("Product"):
            name = product.attrib['specieID']
            r['product'].append(name)

    return species_kdiff, reactions


def reaction_filter(reactions, reactants_left=None, percent_biggest_edges_to_left=None):
    """

    :param reactions:
    :param reactants_left:
    :param percent_biggest_edges_to_left:
        the percent value of biggest edges to left: 0-100. Default is None, meaning - left edges of all nodes selected.
    :return:
    """

    def is_add(name):
        for r in reactants_left:
            if r in name:
                return True
        return False

    if reactants_left:
        filtered = []
        for k, v in reactions.items():
            for r in v['reactant']:
                for p in v['product']:
                    if is_add(r) or is_add(p):
                        filtered.append((k, v))
        reactions = dict(filtered)

    if percent_biggest_edges_to_left:
        p = percent_biggest_edges_to_left/100
        sorted_dict = [x for x in sorted(reactions.items(), key=lambda x: -(x[1]['forward'] if x[1]['forward'] > x[1]['reverse'] else x[1]['reverse']))]
        max_len = round(len(sorted_dict)*p)
        reactions = dict(sorted_dict[:max_len])

    return reactions


def create_graph(reactions, reactants=None, height="100%", width="100%", bgcolor="#222222", font_color="white"):
    g = Network(height=height, width=width, bgcolor=bgcolor, font_color=font_color, directed=True)

    nodes = []
    if reactants is None:
        reactants = []
    for k, v in reactions.items():
        for r in v['reactant']:
            for p in v['product']:
                fr = float(v['forward'])
                rr = float(v['reverse'])
                title = "%s/%s" % (fr, rr)

                if r not in nodes:
                    color = "#f5ce42" if r in reactants else "#80bfff"
                    nodes.append(r)
                    g.add_node(r, color=color)
                if p not in nodes:
                    color = "#f5ce42" if p in reactants else "#80bfff"
                    nodes.append(p)
                    g.add_node(p, color=color)
                if k not in nodes:
                    nodes.append(k)
                    g.add_node(n_id=k, shape="diamond", color="#969696", label="R", size=10, title=k)

                if fr > rr:
                    color = "#91db7b"  # green - domination of forward reaction
                    value = fr
                else:
                    color = "#ff9999"  # red - domination of reverse reaction
                    value = rr

                g.add_edge(r, k, color=color, value=value, title=title)
                g.add_edge(k, p, color=color, value=value, title=title)

    return g


description = """
Creating reaction graph in the web browser.
* square - represents molecule
* diamond - represents reaction
* arrow direction - represents the forward direction
* arrow thickness - represents value of the reaction that represents its color

Edge colors:
* green - represents domination of the forward rate
* red - represents domination of the reverse rate

Square colors:
* blue - regular particle
* yellow - (if selected) key particles from reactants 
"""
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--reaction_file", help="Xml file containing reactions.", required=True)
    ap.add_argument("--result_folder", help="Path to folder where to put HTML result files, default: reaction_visualizations/", default='reaction_visualizations')
    ap.add_argument("--left_edges", help="Left only the percent value of the biggest edges from 0 to 100, default: 100", default=100, type=float)
    ap.add_argument("--node_distance", help="Distance between nodes on the graph, default: 140.", default=140)
    ap.add_argument("--reactants", nargs='+', help="Reduce graph particles only to those denifed here, dfault: None, meaning - left edges of all nodes selected.", default=None)
    args = ap.parse_args()

    species_kdiff, reactions = neurord_parse_reaction_file(filename=args.reaction_file)

    reactions = reaction_filter(reactions, reactants_left=args.reactants, percent_biggest_edges_to_left=args.left_edges)
    graph = create_graph(reactions=reactions, reactants=args.reactants)

    graph.show_buttons(filter_=['physics'])
    graph.hrepulsion(node_distance=args.node_distance, spring_strength=0.001)

    name = '_'.join(args.reactants) if args.reactants else 'ALL_REACTIONS'
    os.makedirs(args.result_folder, exist_ok=True)
    graph.show('%s/%s_%s_percent.html' % (args.result_folder, name, args.left_edges))
