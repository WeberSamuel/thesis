import jsonargparse

class LinkedClass:
    def __init__(self) -> None:
        self. a = 1

class Leaf:
    def __init__(self, a: int, b: int, link: LinkedClass = None) -> None:
        pass
        
class Node:
    def __init__(self, sub_class: Leaf) -> None:
        pass

parser = jsonargparse.ArgumentParser(parser_mode='omegaconf')
# Here add to the parser only argument(s) relevant to the problem
parser.add_subclass_arguments(Node, "Node")
parser.add_class_arguments(LinkedClass, "Linked")
parser.add_argument('--config', action=jsonargparse.ActionConfigFile)
parser.link_arguments("Linked.a", "Node.init_args.sub1_class.init_args.a", apply_on="instantiate")
parser.link_arguments("Linked.a", "Node.init_args.sub1_class.init_args.b", apply_on="instantiate")

# Preferable that the command line arguments are given to the parse_args call
result = parser.parse_args()
result_init = parser.instantiate_classes(result)
print(parser.dump(result))