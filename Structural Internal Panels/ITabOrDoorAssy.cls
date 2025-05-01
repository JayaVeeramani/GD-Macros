VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ITabOrDoorAssy"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Public TabAssy As ILTabAssy
Public Doorassy As IDoorAssy
Public IsDoor As Boolean

Property Get xPoint() As Double

    If IsDoor Then
    
       xPoint = Doorassy.StartComp.xMin
       
    Else
    
        xPoint = TabAssy.LowerLeftXPoint
        
    End If
    
End Property
