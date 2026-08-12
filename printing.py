def print_trail(trail, solver):
    for vs in trail:
        c = 0
        for v in vs[::-1]:
            c <<= 1
            c += solver.Value(v)
        print(hex(c)[2:].zfill(len(vs)//4))

def print_trail2(trail, solver):
    print("\"(",sep="",end="")
    count=0
    for vs in trail:
        c = 0
        for v in vs[::-1]:
            c <<= 1
            c += solver.Value(v)
        if(count<2): print("0x",hex(c)[2:].zfill(len(vs)//4),",",sep="",end="")
        if(count==2): print("0x",hex(c)[2:].zfill(len(vs)//4),")\"",sep="",end="")
        count+=1

def print_trail3(trail, solver):
    for vs in trail:
        c = 0
        for v in vs:
            c <<= 1
            c += solver.Value(v)
        print(hex(c)[2:].zfill(len(vs)//4))

def print_aradi_trail(trail, solver):
    for vs in trail:
        c = 0
        for v in vs:
            if(solver.Value(v)==1):
                print("x", end =" ")
            else:
                print("-", end =" ")
            c+=1
            if(c==32):
                print("")
                c=0
        print(" ")
    