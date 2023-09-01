from dataclasses import dataclass
from jsonargparse import CLI
from src.p2e.networks import RSSMKwargs

def b(c: RSSMKwargs = RSSMKwargs()):
    pass


CLI(b).parse_args()
