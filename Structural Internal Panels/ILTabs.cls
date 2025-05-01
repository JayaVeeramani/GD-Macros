VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ILTabs"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cLoop As SldWorks.Loop2
Private cComp As SldWorks.Component2
Dim HorEdgeList As IArrListObject
Dim VerEdgeList As IArrListObject

Property Get GetLoop() As SldWorks.Loop2
    Set GetLoop = cLoop
End Property

Property Get GetComponent() As SldWorks.Component2
    Set GetComponent = cComp
End Property

Sub Initialize(swLoop As SldWorks.Loop2, swComp As SldWorks.Component2, swView As SldWorks.View)

   Set cLoop = swLoop
   Set cComp = swComp
   Call SegregateHorizontalAndVerticalEdges(swView)

End Sub

Property Get EndHorizontalEdge() As SldWorks.Edge
    
    If Not IsEmpty(HorEdgeList.Items) Then
    
        Set GetEndHorizontalEdge = HorEdgeList.Items(LBound(HorEdgeList.Items)).GetEdge
        
    End If

End Property

Property Get EndVerticalEdge() As SldWorks.Edge
    
    If Not IsEmpty(VerEdgeList.Items) Then
    
        Set EndVerticalEdge = VerEdgeList.Items(LBound(VerEdgeList.Items)).GetEdge
        
    End If

End Property

Property Get IsLeft() As Boolean

    IsLeft = False

    If VerEdgeList.Items(LBound(VerEdgeList.Items)).Point < VerEdgeList.Items(UBound(VerEdgeList.Items)).Point Then
    
        IsLeft = True
        
    End If
    

End Property

Property Get IsBottom() As Boolean

    IsBottom = False

    If HorEdgeList.Items(LBound(HorEdgeList.Items)).Point < HorEdgeList.Items(UBound(HorEdgeList.Items)).Point Then
    
        IsBottom = True
        
    End If

End Property

Property Get xPoint() As Double

    xPoint = Round(VerEdgeList.Items(LBound(VerEdgeList.Items)).Point, 4)

End Property

Property Get yPoint() As Double

    yPoint = Round(HorEdgeList.Items(LBound(HorEdgeList.Items)).Point, 4)

End Property

Private Sub SegregateHorizontalAndVerticalEdges(swView As SldWorks.View)

    Set HorEdgeList = New IArrListObject
    Set VerEdgeList = New IArrListObject
    
    Dim vEdges As Variant
    vEdges = cLoop.GetEdges
    
    Dim i As Integer
    For i = LBound(vEdges) To UBound(vEdges)
    
        Dim swEdge As SldWorks.Edge
        Set swEdge = vEdges(i)
    
        Dim swCurve As SldWorks.Curve
        Set swCurve = swEdge.GetCurve
        
        If swCurve.IsLine Then
        
            Dim vStartVertex As Variant
            vStartVertex = swEdge.GetStartVertex.GetPoint

            Dim vEndVertex As Variant
            vEndVertex = swEdge.GetEndVertex.GetPoint
            
            Dim Length As Double
            Length = GetDistance(vStartVertex, vEndVertex)

            vStartVertex = GetComponentPointInSheetSpace(cComp, vStartVertex, swView)
            vEndVertex = GetComponentPointInSheetSpace(cComp, vEndVertex, swView)
            
            Dim oEdge As IEdgeClass
            Set oEdge = New IEdgeClass

            If Abs(vStartVertex(0) - vEndVertex(0)) <= 0.0001 Then
                
                oEdge.Initialize swEdge, True, CDbl(vStartVertex(0)), Length
                VerEdgeList.AddtoList oEdge
                
            ElseIf Abs(vStartVertex(1) - vEndVertex(1)) <= 0.0001 Then
            
                oEdge.Initialize swEdge, False, CDbl(vStartVertex(1)), Length
                HorEdgeList.AddtoList oEdge
                
            End If
            
        End If

    Next i
    
    HorEdgeList.SortItems "Length"
    VerEdgeList.SortItems "Length"

End Sub
