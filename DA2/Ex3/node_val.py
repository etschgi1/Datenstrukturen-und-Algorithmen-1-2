from dataclasses import dataclass
Mat_Nr = [1,2,0,0,4,2,3,2,7,3]

def main():
    print(f"Input: {Mat_Nr}\nNode-Vals: {vals}")
    #compute indizes a,b,c
    a,b,c = int(Mat_Nr[3]/4)+3,int(Mat_Nr[5]/4)+6,Mat_Nr[6]+1
    print(f"Indizes: a = {a}, b = {b}, c = {c}")

    #Fast_Find:
    N1 = Fast_Find(vals)
    print(N1)

@dataclass
class Set_tree:
    set_nr:int
    set_size:int = 1
    next_ptr:object = None
    def __str__(self) -> str:
        return 

class Fast_Find(object):
    def __init__(self, Node_vals) -> None:
        self.Sets = [Set_tree(val) for val in Node_vals]
    def find(self,Nr:int):
        return self.Sets[Nr].set_nr
    def union(self,S1:Set_tree,S2:Set_tree):
        if S1.set_size>=S2.set_size: #first bigger join second to first
            pass
    def __str__(self) -> str:
        return [s for s in self.Sets]
            

class Fast_Union(object):
    def __init__(self) -> None:
        pass
class Almost_Linear(object):
    def __init__(self) -> None:
        pass

if __name__=="__main__":
    vals = [100+z-10*(i+1) for i,z in enumerate(Mat_Nr)]
    main()