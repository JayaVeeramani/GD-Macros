VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IBlockOut"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cLoop As SldWorks.Loop2
Private cComp As SldWorks.Component2
Private cView As SldWorks.View

Dim HorEdgeList As IArrListObject
Dim VerEdgeList As IArrListObject

Public BalloonLegend As String
Public BalloonId As String

Public LeftBlockOut As IBlockOut
Public RightBlockOut As IBlockOut
Public BottomBlockOut As IBlockOut
Public TopBlockOut As IBlockOut

Property Get GetLoop() As SldWorks.Loop2
    Set GetLoop = cLoop
End Property

Property Get GetComponent() As SldWorks.Component2
    Set GetComponent = cComp
End Property

Sub Initialize(swLoop As SldWorks.Loop2, swComp As SldWorks.Component2, swView As SldWorks.View)

   Set cLoop = swLoop
   Set cComp = swComp
   Set cView = swView
   Call SegregateHorizontalAndVerticalEdges

End Sub

Property Get GetBottomEdge() As IEdgeClass

    If HorEdgeList.Count = 2 Then
        
        Set GetBottomEdge = HorEdgeList.Items(LBound(HorEdgeList.Items))
    
    End If
    
End Property

Property Get Length() As Double

    Length = GetBottomEdge.Length

End Property

Property Get Width() As Double
    
    Width = GetLeftEdge.Length
    
End Property

Property Get GetTopEdge() As IEdgeClass

    If HorEdgeList.Count = 2 Then
        
        Set GetTopEdge = HorEdgeList.Items(UBound(HorEdgeList.Items))
    
    End If
    
End Property

Property Get GetLeftEdge() As IEdgeClass

    If VerEdgeList.Count = 2 Then
        
        Set GetLeftEdge = VerEdgeList.Items(LBound(VerEdgeList.Items))
    
    End If
    
End Property

Property Get GetRightEdge() As IEdgeClass

    If VerEdgeList.Count = 2 Then
        
        Set GetRightEdge = VerEdgeList.Items(UBound(VerEdgeList.Items))
    
    End If
    
End Property


Property Get xMin() As Double

    xMin = VerEdgeList.Items(LBound(VerEdgeList.Items)).Point

End Property

Property Get yMin() As Double

    yMin = HorEdgeList.Items(LBound(HorEdgeList.Items)).Point

End Property

Property Get xMax() As Double

    xMax = VerEdgeList.Items(UBound(VerEdgeList.Items)).Point

End Property

Property Get yMax() As Double

    yMax = HorEdgeList.Items(UBound(HorEdgeList.Items)).Point

End Property

Private Sub SegregateHorizontalAndVerticalEdges()

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
            
            vStartVertex = GetComponentPointInSheetSpace(cComp, vStartVertex, cView)
            vEndVertex = GetComponentPointInSheetSpace(cComp, vEndVertex, cView)
            
            Dim oEdge As IEdgeClass
            Set oEdge = New IEdgeClass

            If Abs(vStartVertex(0) - vEndVertex(0)) <= 0.00001 Then
                
                oEdge.Initialize swEdge, True, CDbl(vStartVertex(0)), Length
                VerEdgeList.AddtoList oEdge
                
            ElseIf Abs(vStartVertex(1) - vEndVertex(1)) <= 0.00001 Then
            
                oEdge.Initialize swEdge, False, CDbl(vStartVertex(1)), Length
                HorEdgeList.AddtoList oEdge
                
            End If
            
        End If

    Next i
    
    HorEdgeList.SortItems "Point", False
    VerEdgeList.SortItems "Point", False

End Sub

Function GetVertexPoint(swEdge As SldWorks.Edge, PointToCheck As Double, Optional IsXVal As Boolean = True) As SldWorks.Vertex
    
    Dim Idx As Integer
    If IsXVal Then
    
        Idx = 0
        
    Else
    
        Idx = 1
        
    End If
    
    Dim swStartVertex As SldWorks.Vertex
    Set swStartVertex = swEdge.GetStartVertex
    
    Dim swEndVertex As SldWorks.Vertex
    Set swEndVertex = swEdge.GetEndVertex
    
    Dim vStartPoint As Variant
    vStartPoint = swStartVertex.GetPoint
    
    Dim vEndPoint As Variant
    vEndPoint = swEndVertex.GetPoint
    
    vStartPoint = GetComponentPointInSheetSpace(cComp, vStartPoint, cView)
    vEndPoint = GetComponentPointInSheetSpace(cComp, vEndPoint, cView)
    
    If Abs(PointToCheck - vStartPoint(Idx)) <= 0.0001 Then
    
        Set GetVertexPoint = swStartVertex

    ElseIf Abs(PointToCheck - vEndPoint(Idx)) <= 0.0001 Then
    
        Set GetVertexPoint = swEndVertex

    End If

End Function
