#現状のベスト
#複合語考慮＋集約と汎化をそれぞれ別別に算出（論文のプログラム）＋最後単純な平均

import heapq
#入力ファイルはpu形式でシンプルに描く　クラス名はwordnet形式にすること
#関連線の多重度には対応している。

class Diagram():#クラス図全体
    def __init__(self,name):
        self.name = name
        self.clist = list()#クラス内のクラスリスト
        self.rlist = list()#クラス内の関係線のリスト
        self.all_dis_dict = dict()

    def add_class(self,c):#クラス追加
        self.clist.append(c)

    def add_relation(self,r):#relation追加
        if r.type == "--":# making c.rlist(reverse)
            for c in self.clist:
                if c == r.toclass :
                    c.normalrlist.append(r.reverse())#normalrlist:関連でつながっているクラス間のrelationlist fromが自分のクラス
                elif c == r.fromclass:
                    c.normalrlist.append(r)

        if r.type != "--":#making childNode
            for c in self.clist:
                if r.toclass == c:
                    c.childrlist.append(r)#子クラス-->自分の関係のrelationlist
                if r.fromclass == c:
                    c.parentrlist.append(r)#自分-->親クラスのrelationlist
        self.rlist.append(r)

    def get_top_classes(self):
        topclist = [] #頂点クラス
        for c in self.clist:
            if len(c.parentrlist) == 0:
                topclist.append(c)
        return topclist

    def get_distance(self,c1,c2):
        if self.all_dis_dict == None:
            self.create_all_dis_dict()
        else:
            return min(self.all_dis_dict[c1.name][c2.name], self.all_dis_dict[c2.name][c1.name])

    def create_all_dis_dict(self):
        for c in self.clist:#ノードリストの作成{}
            self.all_dis_dict[c.name] = {}
            for d in self.clist:
                if c != d:
                    self.all_dis_dict[c.name][d.name] = 100#他のクラスの距離の初期値は100
                else:
                    self.all_dis_dict[c.name][d.name] = 0#自分への距離だけ0
            self.all_dis_dict[c.name] = self.calc_distance(self.all_dis_dict[c.name],0)

    def calc_distance(self,disdict,dis):
        check = True
        for classname in disdict.keys():
            if disdict[classname] == dis:
                for r in self.getClass(classname).normalrlist:
                    if disdict[r.toclass.name]==100:
                        disdict[r.toclass.name] = dis+1
                        check = False#更新されたらtrueにする
                for r in self.getClass(classname).parentrlist:
                    if disdict[r.toclass.name]==100:
                        disdict[r.toclass.name] = dis+1
                        check = False
        
        if check:
            return disdict
        self.calc_distance(disdict,dis+1)
        return disdict     


    def __str__(self):#一覧表示
        return self.name.split("/")[-1].split(".")[0]

    def getClass(self,name):#クラス名からクラスを取得
        for c in self.clist:
            if name == c.name:
                return c
        print("UnknownClassNameError:"+name)
        return None

            

class Class():
    def __init__(self, name):
        self.name = name
        self.namelist = name.split("#")#自分の名前複合語ならlist
        self.normalrlist = []#双方向関連
        self.parentrlist = []#自身からの単方向関連
        self.childrlist = []#relation from childrlist
        
    def __str__(self):
        return "classname:"+self.name

    def get_natural_name(self):
        return self.name.split(".")[0]

class Relation():
    def __init__(self, fromclass, type, toclass):
        self.fromclass = fromclass
        self.type = type
        self.toclass = toclass

    def __str__(self):
        return self.fromclass.name+":"+self.type+":"+self.toclass.name

    def reverse(self):
        return Relation(self.toclass,self.type,self.fromclass)


def read_Pu(file_name):
    d = Diagram(file_name)
    data = open(file_name,"r")
    for line in data:
        line = line.rstrip()
        sdata = line.split()
        if len(sdata)>=1 and sdata[0] == "class":
            d.add_class(Class(sdata[1].strip("{")))
        elif len(sdata)>=2 and "--" in sdata[1]:
            d.add_relation(Relation(d.getClass(sdata[0]),sdata[1],d.getClass(sdata[2])))
        elif len(sdata)>=3 and "--" in sdata[2]:
            d.add_relation(Relation(d.getClass(sdata[0]),sdata[2],d.getClass(sdata[4])))
    d.create_all_dis_dict()
    data.close()
    return d
    