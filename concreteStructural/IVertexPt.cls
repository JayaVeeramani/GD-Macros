VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IVertexPt"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cVertex As SldWorks.Vertex
Private cComp As SldWorks.Component2
Private cView As SldWorks.View
Private cPoint As Variant
Private cViewPoint As Variant

Private orgViewPosition As Variant
Private orgScaleDecimal As Double

Property Get Vertex() As SldWorks.Vertex
    
    Set Vertex = cVertex
    
End Property

Property Get xVal() As Double
    
    xVal = cViewPoint(0)

End Property

Property Get yVal() As Double

    yVal = cViewPoint(1)
    
End Property

Sub Initialize(swVertex As SldWorks.Vertex, swComp As SldWorks.Component2, swView As SldWorks.View)

    Set cVertex = swVertex
    Set cComp = swComp
    Set cView = swView
        
    orgViewPosition = cView.Position
    orgScaleDecimal = cView.ScaleDecimal
    
    cPoint = swVertex.GetPoint
        
    Call GetPointLocationInView
    
End Sub

Private Sub GetPointLocationInView()
    
    cViewPoint = GetComponentPointInSheetSpace(cComp, cPoint, cView)
    
End Sub


Sub CheckForUpdateInPointLocation(swView As SldWorks.View)

    If IsViewChanged(swView) Then

        Set cView = swView
        orgViewPosition = cView.Position
        orgScaleDecimal = cView.ScaleDecimal
        Call GetPointLocationInView
        
    End If

End Sub

Private Function IsViewChanged(swView As SldWorks.View) As Boolean
    
    IsViewChanged = False

    If (swView.Name = cView.Name) Then

        Dim actualViewPosition As Variant
        actualViewPosition = swView.Position
    
        If Not (swView.ScaleDecimal = orgScaleDecimal) Or Not (orgViewPosition(0) = actualViewPosition(0)) _
                Or Not (orgViewPosition(1) = actualViewPosition(1)) Or Not (swView.Angle = cView.Angle) Then

            IsViewChanged = True
     
        End If
            
    Else
            
        IsViewChanged = True
            
    End If
    
End Function
