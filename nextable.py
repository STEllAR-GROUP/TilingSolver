

class Nextable:
    options = []

    def __init__(self, name):
        self.name = name
        self.idx = 0

    def next(self, nodes, my_idx=0, presence=None):
        if presence is None:
            presence = set()
        if self.name not in presence:
            next_val = (self.idx + 1) % len(self.options)
            self.idx = next_val
            if next_val == 0 and my_idx+1 < len(nodes):
                presence.add(self.name)
                return False | nodes[my_idx + 1].next(nodes, my_idx + 1, presence)
            elif next_val == 0 and my_idx+1 == len(nodes):
                return True
        elif my_idx+1 < len(nodes):
            return False | nodes[my_idx + 1].next(nodes, my_idx + 1, presence)
        elif my_idx+1 == len(nodes):
            return True
        else:
            return False
        return False

    def get_idx(self):
        return self.idx

    def get_option(self):
        return self.options[self.idx]
