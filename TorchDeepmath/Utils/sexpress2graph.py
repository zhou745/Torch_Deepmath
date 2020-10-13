import numpy as np

class GNN_tree_node(object):
    def __init__(self,token,index):
        self.token = token
        self.index = index
        self.parent = []
        self.parent_index = []
        self.child = []
        self.child_position = []
        self.child_index = []
    
    def add_parent(self,node):
        self.parent.append(node)

    def add_child(self,node,position):
        self.child.append(node)
        self.child_position.append(position)

    def get_parent(self):
        return(self.parent)

    def get_child(self):
        return(self.child)

class GNN_tree(object):
    def __init__(self,sexp=None):
        self.nodes = []
        self.tokens = []

        self.subexpressions = []
        self.root = None
        if sexp is not None:
            if sexp=="":
               sexp="EMPTY_THM"
            self.root = self.parse_tree(sexp)

    def get_adjcent_maxtrix(self):
        #index layout (x,y)
        #-1 means node[x]->node[y]
        #1 means node[x]<-node[y]
        #0 means no connection

        node_num = len(self.nodes)
        
        adj_np = np.zeros((node_num,node_num),dtype=np.int32)
        #loop through x
        for index in range(node_num):
            node = self.nodes[index]
            adj_np[index,node.child_index]=-1
            adj_np[index,node.parent_index]=1

        return(adj_np)
            
    def get_token(self,sexp):
        start = 1
        token = "" if sexp[0]=="(" else sexp[0]
        end = len(sexp)-1 if sexp[0]=="(" else len(sexp)

        while start<len(sexp) and sexp[start] not in [" ",")"]:
            token+=sexp[start]
            start+=1
        return token,start,end

        
    def get_child(self,sexp_list):
        start = 0   #the 0 and 1 position should be (a
        end = 0
        balance =0
        child_list = []
        record_mod = False

        while start <len(sexp_list):

            if record_mod:
                if sexp_list[end]=="(":
                    balance+=1
                elif sexp_list[end]==")":
                    balance-=1
                if balance == 0:
                    if end == len(sexp_list)-1 or sexp_list[end]==" ":
                        true_end = end+1 if end==len(sexp_list)-1 else end
                        child_list.append(sexp_list[start:true_end])
                        record_mod=False
                        start = end+1
            else:
                if sexp_list[end]!=" ":
                    start = end
                    record_mod=True
                    continue
            end+=1
        return(child_list)

    def parse_tree(self,sexp):
        
        #make sure the order of subexpression is the same as nodes
        self.subexpressions.append(sexp)
        token,start,end = self.get_token(sexp)
        parent = GNN_tree_node(token,len(self.nodes))
        self.nodes.append(parent)
        self.tokens.append(token)
        
        if start<end:
            child = self.get_child(sexp[start:end])
            for posi,c in enumerate(child):
                #sharing node first
                if c in self.subexpressions:
                    index = self.subexpressions.index(c)
                    parent.add_child(self.nodes[index],posi)
                    # parent.child.append(self.nodes[index])
                    self.nodes[index].parent.append(parent)
                    parent.child_index.append(index)
                    self.nodes[index].parent_index.append(parent.index)
                else:
                    child_node = self.parse_tree(c)
                    child_node.parent.append(parent)
                    child_node.parent_index.append(parent.index)
                    parent.add_child(child_node,posi)
                    # parent.child.append(child_node)
                    parent.child_index.append(child_node.index)
        return(parent)


