#data module の読み込み
from  AnimFaceGan.data import  *

if __name__=="__main__":
    #DataLoarderクラスのインスタンス化
    loarder=GetDataLoarder()
    #それぞれのインスタンスを読み込む
    f2b=loarder.create_foward2back()
    b2f=loarder.create_back2foward()
    database=loarder.create_database()

    print(database.__name__())

    #foward →backのデータのセットからゲットの流れ
    print("---  Set & Get RealFaces---")
    f2b.SetData("This is RealFaces!")
    print(b2f.GetData())

    # back →forwardのデータのセットからゲットの流れ
    print("---  Set & Get AnimeFaces---")
    b2f.SetData("This is AnimeFace!")
    print(f2b.GetData())


