VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IConsolidatedList"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cQty As Integer
Public Comp As IComp
 
Property Get Qty() As Integer
 
    Qty = cQty
End Property

Property Let Qty(Val As Integer)

    cQty = Val
    
End Property

Public Sub IncQty()
 
    cQty = cQty + 1
End Sub
 
Private Sub Class_Initialize()
 
    cQty = 1

    
End Sub
