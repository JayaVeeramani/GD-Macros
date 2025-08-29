VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IRebarBends"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cBody As SldWorks.Body2
Private cComp As SldWorks.Component2
Private cCompModel As SldWorks.ModelDoc2
Private cToroidalFace As SldWorks.Face2
Private cView As SldWorks.View


Private cxCenter As Double
Private cyCenter As Double
Private czCenter As Double

Private cxCG As Double
Private cyCG As Double
Private czCG As Double

Private cCenterPoint(2) As Double
Private cCGPoint(2) As Double

Private orgViewPosition As Variant
Private orgScaleDecimal As Double

Sub Initialize(swComp As SldWorks.Component2, swView As SldWorks.View) ' oCutlist As ICutlist,
    
    Set cComp = swComp
    Set cCompModel = ResolveAndGetModelDoc(swComp)
    Set cBody = swComp.GetBody()
    Set cView = swView

    Set cToroidalFace = GetToroidalFace()
    
    If Not cToroidalFace Is Nothing Then
    
        Call GetTorusCenters
        Call GetCGPoint
        Call UpdateSheetMaxMinDimensions(swView)
        
    End If
    
End Sub

Sub GetTorusCenters()
    
    Dim swSurface As SldWorks.Surface
    Set swSurface = cToroidalFace.GetSurface
    
    Dim vTorusParams As Variant
    vTorusParams = swSurface.TorusParams
    
    cCenterPoint(0) = vTorusParams(0)
    cCenterPoint(1) = vTorusParams(1)
    cCenterPoint(2) = vTorusParams(2)
    
End Sub

Sub GetCGPoint()
    
    Dim swModelExt As SldWorks.ModelDocExtension
    Set swModelExt = cCompModel.Extension
    
    Dim swMassProp As SldWorks.MassProperty
    Set swMassProp = swModelExt.CreateMassProperty
    
    Dim vCG As Variant
    vCG = swMassProp.CenterOfMass
    
    cCGPoint(0) = vCG(0)
    cCGPoint(1) = vCG(1)
    cCGPoint(2) = vCG(2)
  
End Sub

Function GetToroidalFace() As SldWorks.Face2

    Dim vFaces  As Variant
    vFaces = cBody.GetFaces
    
    Dim i As Integer
    For i = LBound(vFaces) To UBound(vFaces)
    
        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(i)
        
        Dim swSurface As SldWorks.Surface
        Set swSurface = swFace.GetSurface
        
        If swSurface.IsTorus Then
        
            Set GetToroidalFace = swFace
            Exit Function
            
        End If
    
    Next i
    
End Function

Private Sub UpdateSheetMaxMinDimensions(swView As SldWorks.View)

    Dim CenterPoint As Variant
    CenterPoint = GetComponentPointInSheetSpace(cComp, cCenterPoint, swView)

    Dim CGPoint As Variant
    CGPoint = GetComponentPointInSheetSpace(cComp, cCGPoint, swView)
    
    Call GetMaxMinPoint(CenterPoint(0), CGPoint(0), cxCenter, cxCG)
    Call GetMaxMinPoint(CenterPoint(1), CGPoint(1), cyCenter, cyCG)
    Call GetMaxMinPoint(CenterPoint(2), CGPoint(2), czCenter, czCG)
    
    If Not cSketchSegment Is Nothing Then
    
        If cSketchSegment.GetType = swSketchSegments_e.swSketchLine Then
    
            Dim swSketchLine As SldWorks.SketchLine
            Set swSketchLine = cSketchSegment
            
            Dim vStartPoint As Variant
            vStartPoint = GetPointDataInSheetSpace(swSketchLine.GetStartPoint2, cSketchSegment.GetSketch, cComp, swView)
            
            Dim vEndPoint As Variant
            vEndPoint = GetPointDataInSheetSpace(swSketchLine.GetEndPoint2, cSketchSegment.GetSketch, cComp, swView)
            
            Call GetMaxMinPoint(vStartPoint(0), vEndPoint(0), cxMinSkPt, cxMaxSkPt)
            Call GetMaxMinPoint(vStartPoint(1), vEndPoint(1), cyMinSkPt, cyMaxSkPt)
            
            If Abs(vStartPoint(0) - vEndPoint(0)) <= 0.0001 Then
            
                cIsHorizontal = False
                cMid = vStartPoint(0)
                
        
            ElseIf Abs(vStartPoint(1) - vEndPoint(1)) <= 0.0001 Then
            
                cIsHorizontal = True
                cMid = vStartPoint(1)
                
            End If
            
        End If
    End If
    
End Sub

Sub CheckForUpdateInMaxMinDimensions(swView As SldWorks.View)

    If IsViewChanged(swView) Then
        
        Call UpdateSheetMaxMinDimensions(swView)
        Set cView = swView
        orgScaleDecimal = swView.ScaleDecimal
        orgViewPosition = swView.Position
            
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

