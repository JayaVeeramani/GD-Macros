VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ISubWeldment"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private oWeldBody As IWeldBody
Private cBeforeArrList As IArrListObject
Private cAfterArrList As IArrListObject


Property Get GetBodiesListBeforeThisWeldment() As IArrListObject

    Set GetBodiesListBeforeThisWeldment = cBeforeArrList

End Property
Property Get GetBodiesListAfterThisWeldment() As Variant

    Set GetBodiesListAfterThisWeldment = cAfterArrList

End Property


Property Get GetWeldBody() As IWeldBody

    Set GetWeldBody = oWeldBody

End Property


Sub Initialize(oBody As IWeldBody, ArrList As IArrListObject, Optional IsAfter As Boolean = True)
    
    Set oWeldBody = oBody
    Set cBeforeArrList = New IArrListObject
    Set cAfterArrList = New IArrListObject
    Call AddToArrList(ArrList, IsAfter)
    
End Sub

Sub AddToArrList(ArrList As IArrListObject, IsAfter As Boolean)

    If IsAfter Then
        
        Call cAfterArrList.AddItems(ArrList.Items)
        
    Else
    
        cBeforeArrList.AddItems ArrList.Items
        
    End If
        
End Sub
