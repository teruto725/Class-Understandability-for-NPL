from evaluation import Evaluations
from model import Diagram,Class,Relation,read_Pu

filenames = ["CustomerMerchant.pu",
            "FlightReservationSystem.pu",
            "HospitalManagement.pu",
            "LibraryManagement.pu",
            "monika.pu",
            "OnlineBusReservation.pu",
            "SeminarClassDiagram.pu",
            "SGC.pu",
            "TaskManagement.pu",
            "OrderClassDiagramTemplate.pu"]
#filenames = ["ホテル予約システム1.pu"]
diagrams = list()

for file_name in filenames:
    diagram = read_Pu("experiment/classes/"+file_name)
    diagrams.append(diagram)
    print("ReadIsDone:"+str(diagram))


for diagram in diagrams:
    print("CalcEva:"+str(diagram))
    #print([c.name for c  in diagram.clist])
    #print(diagram.all_dis_dict)
    Evaluations.get("crsu",diagram = diagram)
    Evaluations.get("crsu_wv",diagram = diagram)
    Evaluations.get("cufn",diagram = diagram)
    Evaluations.get("cufn_wv",diagram = diagram)
    Evaluations.get("cufn_v2",diagram = diagram)
    Evaluations.get("cufn_wv",diagram = diagram)
    #print(Evaluations.get("cufn_wv",diagram = diagram))
    #print(Evaluations.get("tops_crsu",diagram,None,None,None).value)
    #print(Evaluations.get("ave_bv",diagram = diagram).value)
    #print(Evaluations.get("ave_bv_wv",diagram=diagram).value)
    #print(Evaluations.get("tops_crsu_wv",diagram = diagram).value)

for eva in Evaluations.evas:
    print(eva.to_csv())