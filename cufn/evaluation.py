from .model import Diagram, Class, Relation
from nltk.corpus import wordnet as wn
from scipy import spatial
import gensim
import nltk
import numpy as np
import pandas as pd
from pathlib import Path
path = Path(__file__).parent   # test.pyのあるディレクトリ
path /= '../w2v-dataset/googlenews.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True,limit=500000)    


class Evaluations():
    evas = list()

    @staticmethod
    def create(name,diagram=None,c1=None,c2=None):
        eva = Evaluation(name,diagram,c1,c2)
        Evaluations.evas.append(eva)

    @staticmethod
    def get(name,diagram=None,c1=None,c2=None):#なかったら作ろう！
        for eva in Evaluations.evas:
            if eva.equals(name,diagram,c1,c2):
                return eva
        Evaluations.create(name,diagram,c1,c2)
        return Evaluations.get(name,diagram,c1,c2)
    
    @staticmethod
    def save_to_file():
        pass
    
    def to_data_frame(diagrams):
        li = list()
        for d in diagrams:
            di = {}
            for eva in Evaluations.evas:
                if eva.diagram is not None and eva.diagram == d:
                    di[eva.name] = eva.value
            li.append(di)
        df = pd.DataFrame([], columns = li[0].keys())
        df = pd.concat([df, pd.DataFrame.from_dict(li)])
        return df
            

class Evaluation():
    def __init__(self,name , diagram=None,c1=None,c2=None):
        self.name = name
        self.value = 0.0
        self.diagram = diagram
        self.c1 = c1
        self.c2 = c2
        if self.name == "sim":
            self.value = self.calc_si(c1,c2)
        elif self.name == "cwsim":
            self.value = self.calc_cwsim(c1,c2)
        elif self.name == "hu":
            self.value = self.calc_hu(c1,c2)
        elif self.name == "pu":
            self.value = self.calc_pu(c1,c2)
        elif self.name == "sp":
            self.value = self.calc_sp(c1)
        elif self.name == "rdu":
            self.value = self.calc_rdu(c1,c2,diagram)
        elif self.name == "bv":
            self.value = self.calc_bv(c1)
        elif self.name == "tops_crsu":
            self.value = self.calc_tops_crsu(diagram)
        elif self.name == "crsu":
            self.value = self.calc_crsu(diagram)
        elif self.name == "ave_bv":
            self.value = self.calc_ave_bv(diagram)
        elif self.name == "cufn":
            self.value = self.calc_cufn(diagram)
        elif self.name == "crsu_wv":
            self.value = self.calc_crsu_wv(diagram)
        elif self.name == "rdu_wv":
            self.value = self.calc_rdu_wv(diagram,c1,c2)
        elif self.name == "cwsim_wv":
            self.value = self.calc_cwsim_wv(c1,c2)
        elif self.name == "sim_wv":
            self.value = self.calc_similarity_wv(c1,c2)
        elif self.name == "cufn_wv":
            self.value = self.calc_cufn_wv(diagram)
        elif self.name == "ave_bv_wv":
            self.value = self.calc_ave_bv_wv(diagram)
        elif self.name == "bv_wv":
            self.value = self.calc_bv_wv(c1)
        elif self.name == "sp_wv":
            self.value = self.calc_sp_wv(c1)
        elif self.name == "tops_crsu_wv":
            self.value = self.calc_tops_crsu_wv(diagram)
        elif self.name == "cufn_ave":
            self.value = self.calc_cufn_ave(diagram)
        elif self.name == "cufn_ave_wv":
            self.value = self.calc_cufn_ave_wv(diagram)


    def equals(self,name,diagram=None,c1=None,c2=None):
        if self.name ==name and self.diagram == diagram and self.c1 == c1 and self.c2 == c2:
            return True
        else:
            return False

    def __str__(self):
        if self.c1 != None:
            if self.c2 != None:
                return (self.name+":"+self.c1.name+":"+self.c2.name+":"+str(self.value))
            else:
                return (self.name+":"+self.c1.name+":"+str(self.value))
        return (self.name+":"+str(self.value))

    def to_dict(self):
        return {"name":self.name,"c1":self.c1,"c2":self.c2,"value":self.value}

    def to_csv(self):
        s = str()
        s += str(self.diagram)+","
        s += self.name+","
        s += self.c1.name+"," if self.c1 != None else "None,"
        s += self.c2.name+"," if self.c2 != None else "None,"
        s += str(self.value)
        return s

    

################################################nirmal

    def calc_similarity(self,cx1,cx2):
        if type(cx1) is Class:
            cx1 = cx1.name
            cx2 = cx2.name
        if type(cx2) is not nltk.corpus.reader.wordnet.Synset:
            cx1 = wn.synset(cx1)
            cx2 = wn.synset(cx2)
        return cx1.path_similarity(cx2)
            
    def calc_cwsim(self,c1,c2):#c1とc2のpathlenghtを算出する
        pllist = []
        for cn1 in c1.namelist:
            for cn2 in c2.namelist:
                pllist.append(self.calc_similarity(cn1,cn2))
        pllist = sorted(pllist, reverse=True)
        if len(pllist) == 4:#上2つの平均
            return (pllist[0]+pllist[1])/2
        else:#最大値を返す
            return pllist[0]

    def calc_hu(self,parent_c,child_c):
        maxhu = 0.0
        for childcn in child_c.namelist:#2重forで総当たり
            childwn = wn.synset(childcn)
            for parentcname in parent_c.namelist:
                parent_wn = wn.synset(parentcname)
                commonhyper=parent_wn.lowest_common_hypernyms(childwn)#listで出てくるので注意list[0]が一番近いてか理想は自分ずれてる時だけずれる
                if parent_wn == commonhyper[0]:#ここの0の決め打ち注意
                    return(1.0)
                hu= self.calc_similarity(parent_wn,commonhyper[0])
                maxhu = max(maxhu,hu)
        return maxhu
    
    def calc_pu(self,whole_c,part_c):#PUを計算する whole_c:全体クラス,c2:部分クラス
        max_pu = 0.0
        for part_cn in part_c.namelist:#部分クラス名を全体語に置き換える
            part_wn = wn.synset(part_cn)
            if part_wn.part_holonyms() !=[]:#集約かつ子クラスが部分語だったら
                part_holo_wns = part_wn.part_holonyms()#全体語に置き換える（ここ書き換えること）
                for part_holo_wn in part_holo_wns:
                    for whole_cn in whole_c.namelist:
                        whole_wn = wn.synset(whole_cn)
                        pu = self.calc_similarity(part_holo_wn,whole_wn)
                        max_pu=max(max_pu,pu)
            else:
                for whole_cn in whole_c.namelist:
                    whole_wn = wn.synset(whole_cn)
                    pu = self.calc_similarity(part_wn,whole_wn)
                    max_pu=max(max_pu,pu)
        return max_pu
    
    def calc_sp(self,top_c):#spの計算式 c1 = topclass
        for i in range(len(top_c.childrlist)-1):
            achild = wn.synset(top_c.childrlist[i].fromclass.namelist[0])
            for j in range(i+1,len(top_c.childrlist)):
                bchild = wn.synset(top_c.childrlist[j].fromclass.namelist[0])
                sp = 1.0-abs(self.calc_similarity(wn.synset(top_c.name),achild)-self.calc_similarity(wn.synset(top_c.name),bchild))
        return sp

    def calc_rdu(self,c1,c2,diagram):
        if diagram.get_distance(c1,c2) != 100:#経路がつながっていたら
            CWsim = Evaluations.get("cwsim",c1 = c1,c2 = c2)
            #print(CWsim)
            Esim = 1.0/(diagram.get_distance(c1,c2)+1)
            #print(Esim)
            RDU = 1.0 - abs(CWsim.value-Esim)
            return RDU
        else:
            return None

    def calc_bv(self,top_c):#blockvalue
        evas = list()
        if top_c.childrlist:
            for childr in top_c.childrlist:
                if childr.type == "--|>":
                    hu = Evaluations.get("hu",c1 = top_c,c2 = childr.fromclass)
                    evas.append(hu.value)
                elif childr.type == "--*":
                    pu = Evaluations.get("pu",c1=top_c,c2=childr.fromclass)
                    evas.append(pu.value)
            if len(top_c.childrlist) >=2:#子クラスを2つ以上持つとき
                sp = Evaluations.get("sp",c1=top_c)
                evas.append(sp.value)
            return sum(evas)/len(evas)
        else:
            return None

    def calc_tops_crsu(self,diagram):
        top_classes = diagram.get_top_classes()
        rdus = list()
        for i in range(len(top_classes)-1):
            for j in range(i+1,len(top_classes)):
                rdu =  Evaluations.get("rdu",diagram=diagram,c1 = top_classes[i],c2=top_classes[j]).value
                if rdu is not None:
                    rdus.append(rdu)
        if len(rdus) == 0:
            return 0
        return sum(rdus)/len(rdus)

    def calc_crsu(self,diagram):
        rdus = list()
        cl = diagram.clist
        for i in range(len(cl)-1):
            for j in range(i+1,len(diagram.clist)):
                rdu =  Evaluations.get("rdu",diagram = diagram,c1 = cl[i],c2 = cl[j])
                if rdu.value is not  None:
                    rdus.append(rdu.value)
                    #print(rdu)
        if len(rdus) == 0:
            return None
        return sum(rdus)/len(rdus)

    def calc_ave_bv(self,diagram):
        evas_bv = list()
        for top_c in diagram.get_top_classes():
            if Evaluations.get("bv",c1 = top_c).value is not None:
                evas_bv.append(Evaluations.get("bv",c1 = top_c).value)
        if len(evas_bv) == 0:
            return 0
        return sum(evas_bv)/len(evas_bv)

    def calc_cufn(self,diagram):
        ave_bv = Evaluations.get("ave_bv",diagram=diagram).value
        tops_crsu = Evaluations.get("tops_crsu",diagram=diagram).value
        return ave_bv +tops_crsu

################ crsu
    def calc_crsu_wv(self,diagram):
        rdus = list()
        cl = diagram.clist
        for i in range(len(cl)-1):
            for j in range(i+1,len(diagram.clist)):
                rdu =  Evaluations.get("rdu_wv",diagram = diagram,c1 = cl[i],c2 = cl[j])
                if rdu.value is not  None:
                    rdus.append(rdu.value)
        
        if len(rdus) == 0:
            return None
        return sum(rdus)/len(rdus)

    def calc_rdu_wv(self,diagram,c1,c2):
        if diagram.get_distance(c1,c2) != 100:#経路がつながっていたら
            CWsim = Evaluations.get("cwsim_wv",c1 = c1,c2 = c2)
            #print(CWsim)
            Esim = 1.0/(diagram.get_distance(c1,c2)+1)
            #print(Esim)
            RDU = 1.0 - abs((CWsim.value)-Esim)
            return RDU
        else:
            return None

    def calc_cwsim_wv(self,c1,c2):#c1とc2のpathlenghtを算出する
        pllist = []
        for cn1 in c1.namelist:
            for cn2 in c2.namelist:
                pllist.append(self.calc_similarity_wv(cn1,cn2))
        pllist = sorted(pllist, reverse=True)
        if len(pllist) == 4:#上2つの平均
            return (pllist[0]+pllist[1])/2
        else:#最大値を返す
            return pllist[0]
    
    def calc_similarity_wv(self,w1,w2):
        w1 = w1.split(".")[0].split("_")
        w2 = w2.split(".")[0].split("_")
        return 1 - spatial.distance.cosine(self.avg_vec(w1), self.avg_vec(w2))
   
    def avg_vec(self,wlist):#ベクトル平均
        sumvec = model[wlist.pop(0)]
        if len(wlist) == 0:
            return sumvec
        for word in wlist:
            sumvec = np.add( sumvec,model[word])
        return np.divide(sumvec,len(wlist))

############## wvecのcufn
    def calc_cufn_wv(self,diagram):
        ave_bv_wv = Evaluations.get("ave_bv_wv",diagram=diagram).value
        tops_crsu_wv = Evaluations.get("tops_crsu_wv",diagram=diagram).value
        return ave_bv_wv +tops_crsu_wv

    def calc_tops_crsu_wv(self,diagram):
        top_classes = diagram.get_top_classes()
        rdus = list()
        for i in range(len(top_classes)-1):
            for j in range(i+1,len(top_classes)):
                rdu =  Evaluations.get("rdu_wv",diagram=diagram,c1 = top_classes[i],c2=top_classes[j]).value
                if rdu is not None:
                    rdus.append(rdu)
        if len(rdus) == 0:
            return 0
        return sum(rdus)/len(rdus)

    def calc_ave_bv_wv(self,diagram):
        evas_bv = list()
        for top_c in diagram.get_top_classes():
            if Evaluations.get("bv_wv",c1 = top_c).value is not None:
                evas_bv.append(Evaluations.get("bv_wv",c1 = top_c).value)
        if len(evas_bv) == 0:
            return 0
        return sum(evas_bv)/len(evas_bv)
    
    def calc_bv_wv(self,top_c):
        evas = list()
        if top_c.childrlist:
            for childr in top_c.childrlist:
                if childr.type == "--|>":
                    hu = Evaluations.get("hu",c1 = top_c,c2 = childr.fromclass)
                    evas.append(hu.value)
                elif childr.type == "--*":
                    pu = Evaluations.get("pu",c1=top_c,c2=childr.fromclass)
                    evas.append(pu.value)
            if len(top_c.childrlist) >=2:#子クラスを2つ以上持つとき
                sp = Evaluations.get("sp_wv",c1=top_c)
                evas.append(sp.value)
            return sum(evas)/len(evas)
        else:
            return None
    
    def calc_sp_wv(self,top_c):
        for i in range(len(top_c.childrlist)-1):
            achild = top_c.childrlist[i].fromclass.namelist[0]
            for j in range(i+1,len(top_c.childrlist)):
                bchild = top_c.childrlist[j].fromclass.namelist[0]
                sp = 1.0-abs(self.calc_similarity_wv(top_c.name,achild)-self.calc_similarity_wv(top_c.name,bchild))
        return sp

###calc###########################
    def calc_num_relations(self,diagram):
        pass



########cufuを個数で平均とる方式に変更
    def calc_cufn_ave(self,diagram):
        top_classes = diagram.get_top_classes()
        rdus = list()
        for i in range(len(top_classes)-1):
            for j in range(i+1,len(top_classes)):
                rdu =  Evaluations.get("rdu",diagram=diagram,c1 = top_classes[i],c2=top_classes[j]).value
                if rdu is not None:
                    rdus.append(rdu)
        if len(rdus) == 0:
            return 0
        evas = list()
        if top_c.childrlist:
            for childr in top_c.childrlist:
                if childr.type == "--|>":
                    hu = Evaluations.get("hu",c1 = top_c,c2 = childr.fromclass)
                    evas.append(hu.value)
                elif childr.type == "--*":
                    pu = Evaluations.get("pu",c1=top_c,c2=childr.fromclass)
                    evas.append(pu.value)
            if len(top_c.childrlist) >=2:#子クラスを2つ以上持つとき
                sp = Evaluations.get("sp",c1=top_c)
                evas.append(sp.value)
            return sum(evas)/len(evas)
        else:
            return None
    def calc_cufn_ave_wv(self,diagram):
        pass