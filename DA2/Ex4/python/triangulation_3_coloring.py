EDGES = [("5-6","4,7"),("1-2","3"),("1-3","2,5"),("1-5","3"),("2-3","1,4"),("2-4","3"),("3-4","2,5"),("3-5","1,4"),("4-5","3,6"),("4-6","5"),("5-7","6"),("6-7","5"),] #Can be colored
FAULTYEDGES = [("1-3","2,4"),("1-2","3"),("1-4","3"),("2-3","1,4"),("2-4","3,5"),("2-5","4"),("3-4","1,2"),("4-5","2")] #can not be colored
FAULTYEDGESM = [("1-2","3"),("1-3","2,4"),("1-4","3"),("2-3","1,4"),("2-4","3"),("3-4","1,2")] #modified without edges to 5
EDGE_DIC = {}
COLORS = []


def get_points(edge_trig):
    edge, trig_points = edge_trig
    return ([int(x) for x in edge if x.isnumeric()],trig_points)

def colorin(edge,first=False): #first for first edge
    global COLORS,EDGE_DIC
    colnum = 0
    edge, trig_points = get_points(edge)
    if first:
        COLORS[edge[0]-1]=1
        COLORS[edge[1]-1]=2
        colnum = 2
    Cols = [COLORS[edge[0]-1],COLORS[edge[1]-1]]
    if Cols[0]==Cols[1]:
        raise Exception("Not colorable")
    for p in trig_points:
        colpoint = COLORS[p-1]
        if colpoint in Cols: #something went wrong
            raise Exception("Not colorable")
        elif colpoint == 0: #not already colored
            col = [x for x in range(1,4) if x not in Cols][0]
            COLORS[p-1] = col
            colnum+=1
    del EDGE_DIC[f"{edge[0]}-{edge[1]}"] #delete cur edge from dic, 
    return colnum
    
#info bezüglich del EDGE_DIC eine flag hier wäre kluger, dann könnte ma diese auch bei get_next_edges setzen und hätte nicht das Problem, dass eine Edge schon delete wurde von get_next_edges, man könnte sich dann den get_next_edges toadd if check sparen aber tbh wurscht (wäre effizienter klarerweise)
            
def get_next_edges(cur_edge, edge_stack):
    global COLORS,EDGE_DIC
    """Retrun all edges which are in EDGE_DIC and which start at triangle point and end in other point from edge
    Min 0 (if no left), Max 4 for inner edge

    Args:
        cur_edge ([type]): [description]
    """
    edge, trig_points = get_points(cur_edge)
    pot_edges = []
    for i in range(len(trig_points)):
        for j in range(2):
            pedge = f"{min([trig_points[i],edge[j]])}-{max([trig_points[i],edge[j]])}"
            try:
                toadd = (pedge,EDGE_DIC[pedge])
                if toadd not in edge_stack:
                    pot_edges.append(toadd)
            except KeyError:
                continue
    return pot_edges
def reset():
    global COLORS,EDGE_DIC
    EDGE_DIC = {}
    COLORS = []
def color(Edge_list):
    global COLORS,EDGE_DIC
    reset()
    #setup get max from EDGE_Triangle Points
    max_ = 0
    for c,t in enumerate(Edge_list):
        back = [int(x) for x in t[1] if x.isnumeric()]
        EDGE_DIC[t[0]] = back
        Edge_list[c] = (t[0],back) 
        max_ = max(back) if max(back)>max_ else max_
    COLORS = [0]*max_
    print(max_)
    counter = max_ #remaining
    # COLORS = [0]*max_
    first_edge = Edge_list.pop(0)
    #info color first edge
    counter-= colorin(first_edge, first=True)
    #info loop:
    edge_stack = []
    # counter = max_ - sum(1 for c in COLORS if c is not 0)
    cur_edge = first_edge
    while counter != 0:
        edge_stack += get_next_edges(cur_edge,edge_stack)
        if edge_stack:
            cur_edge = edge_stack.pop(0)
            counter -= colorin(cur_edge)
        else:
            raise Exception("Edge_stack empty!")
    #check rest of edges:
    for k,v in EDGE_DIC.items():
        if (k,v) not in edge_stack:
            edge_stack.append((k,EDGE_DIC[k]))
    for edge in edge_stack:
        tocheck,_ = get_points(edge)
        if COLORS[tocheck[0]-1] == COLORS[tocheck[1]-1]:
            raise Exception("Not colorable")
    RBG = {1:"R",2:"B",3:"G"}
    print(f"Done Coloring: {[(i+1,RBG[c]) for i,c in enumerate(COLORS)]}")


# color(EDGES)
color(FAULTYEDGES)
# color(FAULTYEDGESM)
