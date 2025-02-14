class LazyList:
    def __init__(self, function, length):
        self.function = function
        self.length = length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError("List index out of range")
            
        return self.function(index)
    
    def __len__(self):
        return self.length
    

class DynamicConnectivity:
    def __init__(self, n):
        self.n = n
        self.parent = [i for i in range(n)]
        self.size = [1 for _ in range(n)]

    def find(self, p):
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return

        if self.size[rootP] < self.size[rootQ]:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        else:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def get_connected_components(self):
        component_map = {}
        
        for i in range(self.n):
            root = self.find(i)
            if root not in component_map:
                component_map[root] = list()
                component_map[root].append(i)
            else:
                component_map[root].append(i)
        components = list(component_map.values())
        return components


class ConstantLengthList:
    def __init__(self, max_length):
        self.index = 0
        self.list = list()
        self.max_length = max_length

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < len(self.list):
            self.index += 1
            return self.list[self.index - 1]
        else:
            raise StopIteration
    
    def __getitem__(self, index):
        return self.list[index]
    
    def __setitem__(self, index, value):
        self.list[index] = value
    
    def __len__(self):
        return len(self.list)

    def append(self, item):
        self.list.append(item)
        if len(self.list) > self.max_length:
            self.list.pop(0)
    
    def pop(self):
        return self.list.pop()