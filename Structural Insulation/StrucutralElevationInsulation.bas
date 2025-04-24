Attribute VB_Name = "StrucutralElevationInsulation"
Public swApp As SldWorks.SldWorks
Public swTopLevelModel As SldWorks.ModelDoc2
Public swConfig As SldWorks.Configuration
Public IsSubAssyFormClicked As Boolean
Public IsInsulationFormClicked As Boolean

Public swMathUtility As SldWorks.MathUtility

Sub main()

    Set swApp = Application.SldWorks
    Set swTopLevelModel = swApp.ActiveDoc
    
    If swTopLevelModel.GetType = swDocumentTypes_e.swDocASSEMBLY Then
    
        Set swMathUtility = swApp.GetMathUtility
        
        Call PopulateDisplayStateInForm
        WallDrawingForm.ProjectNoBox.Value = Mid(swTopLevelModel.GetPathName, InStrRev(swTopLevelModel.GetPathName, "\") + 1, 6)
        WallDrawingForm.Show vbModeless

    End If
    
End Sub

Function PopulateDisplayStateInForm()

    Set swConfig = swTopLevelModel.ConfigurationManager.ActiveConfiguration
    
    Dim vDisplayStateNames As Variant
    vDisplayStateNames = swConfig.GetDisplayStates()
    
    Dim i As Integer
    For i = LBound(vDisplayStateNames) To UBound(vDisplayStateNames)
        
        With WallDrawingForm.DisplayList
        
            .AddItem
            .List(i) = vDisplayStateNames(i)
            
        End With
        
    Next i
    
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
    
    Dim XForm As SldWorks.MathTransform
    Set XForm = swComp.Transform2.Multiply(swView.ModelToViewTransform)
    Set XForm = XForm.Multiply(swSketch.ModelToSketchTransform)
    
    GetComponentPointInViewSpace = GetTransformPoint(vPoint, XForm)

End Function

            


