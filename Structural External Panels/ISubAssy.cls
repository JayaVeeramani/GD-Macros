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

Public BottomEdge As SldWorks.Edge

Public StartIdx As Integer
Public EndIdx As Integer

Public StartComp As IComp
Public EndComp As IComp

Public Dimension As SldWorks.DisplayDimension
Public AssyLength As Double

Private DoorOrHVACAssyList As IArrListObject
'Private ContourCutlist As IArrListObject
Private HVACAssyList As IArrListObject
Private DoorAssyList As IArrListObject
Private cDoorArea As Double

Property Get GetDoorOrHVACAssemblies() As Variant
    GetDoorOrHVACAssemblies = DoorOrHVACAssyList.Items
End Property

Property Get GetDoorAssemblies() As Variant
    GetDoorAssemblies = DoorAssyList.Items
End Property

Property Get GetHVACAssemblies() As Variant
    GetHVACAssemblies = HVACAssyList.Items
End Property

'Property Get GetContourSketches() As Variant
'    GetContourSketches = ContourCutlist.Items
'End Property
Function GetMaxLength() As Double

    Dim yMin As Double
    Dim yMax As Double
    
    If StartComp.yMin < EndComp.yMin Then
    
        yMin = StartComp.yMin
    
    Else
        
        yMin = EndComp.yMin
        
    End If
    
    If StartComp.yMax > EndComp.yMax Then
    
        yMax = StartComp.yMax
    
    Else
    
        yMax = EndComp.yMax
        
    End If
    
    GetMaxLength = yMax - yMin

End Function



Property Get TotalDoorArea() As Double

    If IsEmpty(DoorAssyList.Items) Then
        TotalDoorArea = 0
    Else
        If cDoorArea = 0 Then
            
           Call GetDoorArea
            
        End If
        
        TotalDoorArea = cDoorArea

    End If
    
End Property

Sub AddDoororHVACList(DoorOrHVACList As IArrListObject)

    Dim i As Integer
    Dim vDoorOrHVACItems As Variant
    vDoorOrHVACItems = DoorOrHVACList.Items
    
    If Not IsEmpty(vDoorOrHVACItems) Then
    
    For i = LBound(vDoorOrHVACItems) To UBound(vDoorOrHVACItems)
    
        Dim oDoorOrHVACAssy As IDoorOrHVACAssy
        Set oDoorOrHVACAssy = vDoorOrHVACItems(i)
        
        Call AddDoororHVACAssy(oDoorOrHVACAssy)

    Next i
    
    End If
    
End Sub

'Sub AddContourList(ContourList As IArrListObject)
'
'    Dim i As Integer
'
'    Dim vContourItems As Variant
'    vContourItems = ContourList.Items
'
'    If Not IsEmpty(vContourItems) Then
'
'        For i = LBound(vContourItems) To UBound(vContourItems)
'
'            Dim oContour As IContourSketch
'            Set oContour = vContourItems(i)
'
'            ContourCutlist.AddtoList oContour
'
'        Next i
'
'    End If
'
'End Sub
'
'Sub AddContourSketch(oContour As IContourSketch)
'
'    ContourCutlist.AddtoList oContour
'
'End Sub

Sub AddDoororHVACAssy(doororHVACAssy As IDoorOrHVACAssy)
    
    DoorOrHVACAssyList.AddtoList doororHVACAssy
    
    If doororHVACAssy.IsDoor Then
    
        DoorAssyList.AddtoList doororHVACAssy
        
    Else
    
        HVACAssyList.AddtoList doororHVACAssy
        
    End If
        
End Sub

Private Sub GetDoorArea()
    
    Dim vDoorAssy As Variant
    vDoorAssy = DoorAssyList.Items
    
    Dim i As Integer
    For i = LBound(vDoorAssy) To UBound(vDoorAssy)
        
        Dim oDoorAssy As IDoorOrHVACAssy
        Set oDoorAssy = vDoorAssy(i)
        
        cDoorArea = cDoorArea + (oDoorAssy.DoororHVACLength * oDoorAssy.DoororHVACWidth)
    
    Next i
    
End Sub

Private Sub Class_Initialize()

    cDoorArea = 0
    Set DoorOrHVACAssyList = New IArrListObject
    Set HVACAssyList = New IArrListObject
    Set DoorAssyList = New IArrListObject
    Set ContourCutlist = New IArrListObject

End Sub
