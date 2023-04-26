class A:
    def __init__(self, a: int, b: int) -> None:
        pass
        
class Recursive:
    def __init__(self, sub_class: 'A|Recursive') -> None:
        pass

class LinkedClass:
    def __init__(self) -> None:
        self.a = 1
