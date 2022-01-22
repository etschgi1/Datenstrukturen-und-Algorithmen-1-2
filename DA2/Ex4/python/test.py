from itertools import count
from unicodedata import numeric


EDGES = [("1-2","3"),("1-3","2,5"),("1-5","3"),("2-3","1,4"),("2-4","3"),("3-4","2,5"),("3-5","1,4"),("4-5","3,6"),("4-6","5"),("5-6","4,7"),("5-7","6"),("6-7","5"),] #Can be colored
EDGE_DIC = {}
COLORS = {}
def get_points(edge_trig):
    edge, trig_points = edge_trig
    return ([int(x) for x in edge if x.isnumeric()],trig_points)

def colorin(edge,first=False): #first for first edge
    edge, trig_points = get_points(edge)
    if not first:
        Cols = [COLORS[edge[0]],COLORS[edge[1]]]
        if Cols[0]==Cols[1]:
            raise Exception("Not colorable")
        for p in trig_points:
            try:#see if colored
                colpoint = COLORS[p]
                if colpoint in Cols:
                    raise Exception("Not colorable")
            except KeyError: #color point with right color
                col = [x for x in range(1,4) if x not in Cols][0]
                COLORS[p] = col
        del EDGE_DIC[edge[0]] #delete cur edge from dic
    #color first edge
    else:
        COLORS[edge[0]]=1
        COLORS[edge[1]]=2
        
            
def get_next_edges(cur_edge):
    """Retrun all edges which are in EDGE_DIC and which start at triangle point and end in other point from edge
    Min 0 (if no left), Max 4 for inner edge

    Args:
        cur_edge ([type]): [description]
    """
    pass


def color(Edge_list):
    #setup get max from EDGE_Triangle Points
    max_ = 0
    for c,t in enumerate(Edge_list):
        back = [int(x) for x in t[1] if x.isnumeric()]
        EDGE_DIC[t[0]] = back
        Edge_list[c] = (t[0],back) 
        max_ = max(back) if max(back)>max_ else max_
    print(max_)
    counter = max_ #remaining
    # COLORS = [0]*max_
    first_edge = Edge_list.pop()
    #info color first edge
    colorin(first_edge, first=True)
    
    #info loop:
    edge_stack = []
    counter = max_ - len(COLORS.keys()) #little cheat but not relevant can count down manually
    cur_edge = first_edge
    while counter != 0:
        colorin(cur_edge)
        edge_stack.append()
        counter = max_ - len(COLORS.keys())
    

color(EDGES)