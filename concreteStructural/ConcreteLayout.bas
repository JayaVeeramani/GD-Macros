Attribute VB_Name = "ConcreteLayout"
Public swApp As SldWorks.SldWorks
Public swTopLevelModel As SldWorks.ModelDoc2

Public swConcretePanel As SldWorks.Component2

Public swConfig As SldWorks.Configuration
Public swMathUtility As SldWorks.MathUtility

Public Const BalloonWidth As Double = 0.0065
Public Const SheetBorderTop As Double = 0.27030866
Public Const SheetBorderLeft As Double = 0.01590679
Public Const SheetBorderRight As Double = 0.41595679

Public Const RebarTableTemplate As String = "C:\FBD\COMMON\FBD Templates\CUTLIST REBARS.sldwldtbt"
Public Const FoamTableTemplate As String = "C:\FBD\COMMON\FBD Templates\CUTLIST FOAM.sldwldtbt"

Sub main()

    Set swApp = Application.SldWorks
    Set swTopLevelModel = swApp.ActiveDoc
    
    If swTopLevelModel.GetType = swDocumentTypes_e.swDocASSEMBLY Then

        HideShowForm.WeldNoBox.Value = Mid(swTopLevelModel.GetPathName, InStrRev(swTopLevelModel.GetPathName, "\") + 1, 6)
        HideShowForm.Show vbModeless

    End If
    
End Sub

Function ResolveAndGetModelDoc(swComp As SldWorks.Component2) As SldWorks.ModelDoc2

    If swComp.GetSuppression = swComponentSuppressionState_e.swComponentLightweight Then

        Dim bRet As Integer
        bRet = swComp.SetSuppression2(swComponentSuppressionState_e.swComponentResolved)

    End If
    
    Set ResolveAndGetModelDoc = swComp.GetModelDoc2()
    
End Function

Sub GetMaxMinPoint(LowerBoundPoint As Variant, HigherBoundPoint As Variant, _
        ByRef OutLower As Double, ByRef OutHigher As Double)

    If LowerBoundPoint < HigherBoundPoint Then
    
        OutLower = LowerBoundPoint
        OutHigher = HigherBoundPoint
        
    Else

        OutLower = HigherBoundPoint
        OutHigher = LowerBoundPoint
        
    End If
    
End Sub

Function GetComponentPointInSheetSpace(swComp As SldWorks.Component2, _
                vPoint As Variant, swView As SldWorks.View)
    
    GetComponentPointInSheetSpace = GetTransformPoint(vPoint, _
                                swComp.Transform2.Multiply(swView.ModelToViewTransform))

End Function

Function GetTransformPoint(vPoint As Variant, swTransform As SldWorks.MathTransform)
    
    Dim swMathPoint As SldWorks.MathPoint
    Set swMathPoint = swMathUtility.CreatePoint(vPoint)
    
    Set swMathPoint = swMathPoint.MultiplyTransform(swTransform)
    GetTransformPoint = swMathPoint.ArrayData

End Function

Function GetSketchPointInSheetSpace(swView As SldWorks.View, vPoint As Variant)

    Dim swSketch As SldWorks.Sketch
    Set swSketch = swView.GetSketch
    
    GetSketchPointInSheetSpace = GetTransformPoint(vPoint, swSketch.ModelToSketchTransform.Inverse)

End Function

Function GetComponentPointInViewSpace(swComp As SldWorks.Component2, _
                    vPoint As Variant, swView As SldWorks.View)
    
    Dim swSketch As SldWorks.Sketch
    Set swSketch = swView.GetSketch
    
    Dim xForm As SldWorks.MathTransform
    Set xForm = swComp.Transform2.Multiply(swView.ModelToViewTransform)
    Set xForm = xForm.Multiply(swSketch.ModelToSketchTransform)
    
    GetComponentPointInViewSpace = GetTransformPoint(vPoint, xForm)

End Function

Function GetSheetPointInViewSpace(swView As SldWorks.View, vPoint As Variant)

    Dim swSketch As SldWorks.Sketch
    Set swSketch = swView.GetSketch
    
    GetSheetPointInViewSpace = GetTransformPoint(vPoint, swSketch.ModelToSketchTransform)

End Function

Function GetDistance(vStartPt As Variant, vEndPt As Variant) As Double
    
    GetDistance = Sqrt((vStartPt(0) - vEndPt(0)) ^ 2 + (vStartPt(1) - vEndPt(1)) ^ 2 + (vStartPt(2) - vEndPt(2)) ^ 2)

End Function

Function GetEdgeLength(swEdge As SldWorks.Edge) As Double

    Dim vStartVertex As Variant
    vStartVertex = swEdge.GetStartVertex.GetPoint

    Dim vEndVertex As Variant
    vEndVertex = swEdge.GetEndVertex.GetPoint

    GetEdgeLength = GetDistance(vStartVertex, vEndVertex)
            
End Function

Function GetPointDataInSheetSpace(swPoint As SldWorks.sketchPoint, swSketch As SldWorks.Sketch, _
            swComp As SldWorks.Component2, swView As SldWorks.View)

    Dim dPoint(2) As Double
    dPoint(0) = swPoint.X
    dPoint(1) = swPoint.Y
    dPoint(2) = swPoint.Z
    
    Dim vPoint As Variant
    vPoint = dPoint
    
    vPoint = GetTransformPoint(vPoint, swSketch.ModelToSketchTransform.Inverse)
    GetPointDataInSheetSpace = GetComponentPointInSheetSpace(swComp, vPoint, swView)
    
    
End Function


