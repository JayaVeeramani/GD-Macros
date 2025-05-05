VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ISubAssy"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Public StartEdge As SldWorks.Edge
Public EndEdge As SldWorks.Edge
 
Public StartIdx As Integer
Public EndIdx As Integer
 
Public StartComp As IComp
Public EndComp As IComp
 
Public Dimension As SldWorks.DisplayDimension
Private lTaborDoorList As IArrListObject
Private lTabList As IArrListObject

Property Get GetLTabList() As Variant

    GetLTabList = lTabList.Items
    
End Property

Property Get GetLTabOrDoorList() As Variant

    lTaborDoorList.SortItems "xPoint", False
    GetLTabOrDoorList = lTaborDoorList.Items
    
End Property
Property Get yMin() As Double

    If Not StartComp Is Nothing And Not EndComp Is Nothing Then
    
        If StartComp.yMin < EndComp.yMin Then
            yMin = StartComp.yMin
        Else
            yMin = EndComp.yMin
        End If
        
    End If
    
End Property

Property Get yMax() As Double

    If Not StartComp Is Nothing And Not EndComp Is Nothing Then
    
        If StartComp.yMax > EndComp.yMax Then
            yMax = StartComp.yMax
        Else
            yMax = EndComp.yMax
        End If
        
    End If
    
End Property

 
Sub AddEntireList(ArrList As IArrListObject)
    
    Dim vItems As Variant
    vItems = ArrList.Items
    
    If Not IsEmpty(vItems) Then
    
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
            
            Dim oLTaborDoorAssy As Object
            Set oLTaborDoorAssy = vItems(i)
            
            Call AddToLTaborDoorList(oLTaborDoorAssy)
            
        Next i
        
    End If

End Sub
 
Sub AddToLTaborDoorList(oLTaborDoorAssy As Object)
    
    Dim Obj As ITabOrDoorAssy
    Set Obj = New ITabOrDoorAssy
    
    Obj.IsDoor = oLTaborDoorAssy.IsDoor
    
    If oLTaborDoorAssy.IsDoor Then
        
        Set Obj.Doorassy = oLTaborDoorAssy
        
    Else
    
        lTabList.AddtoList oLTaborDoorAssy
        Set Obj.TabAssy = oLTaborDoorAssy
    
    End If
    
    lTaborDoorList.AddtoList Obj

End Sub
 

Private Sub Class_Initialize()
    
    Set lTaborDoorList = New IArrListObject
    Set lTabList = New IArrListObject

End Sub
