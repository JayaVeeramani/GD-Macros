Attribute VB_Name = "AddDimensions"

Private Sub AddQtyBracketsAndSuffixToDimension(swDisplayDim As SldWorks.DisplayDimension, Qty As Integer, Optional IsBrackets As Boolean, Optional suffixNote As String = "")

    If IsBrackets Then
    
         swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, "("
    
        If Qty > 1 Then
            
            swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, Qty & "X ("
                
        End If
        
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextSuffix, ")" & vbCrLf & suffixNote
        
    Else
        
        If Qty > 1 Then
            
            swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, Qty & "X "
                
        End If
        
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextSuffix, vbCrLf & suffixNote
        
    End If


End Sub

Function SelectAndAddOrdinateOrigin(swEnt As SldWorks.Entity, swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, _
        xPos As Double, yPos As Double, Optional IsHorizontal As Boolean = False) As SldWorks.DisplayDimension

    swDrawing.ClearSelection2 True
    swDrawing.SetPickMode
    swView.SelectEntity swEnt, False
    
    If IsHorizontal Then
    
        swDrawing.InsertHorizontalOrdinate

    Else
    
         swDrawing.InsertVerticalOrdinate
         
    End If

    Call swDrawing.Extension.SelectByID2("", "VIEW", xPos, yPos, 0, False, 0, Nothing, 0)
    
    Dim swSelectMgr As SldWorks.SelectionMgr
    Set swSelectMgr = swDrawing.SelectionManager

    Set SelectAndAddOrdinateOrigin = swSelectMgr.GetSelectedObject6(1, -1)

End Function

Sub AddToOrdinateDimension(OrdDim As SldWorks.DisplayDimension, _
                Qty As Integer, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim PrevDimCount As Integer
    PrevDimCount = swView.GetDisplayDimensionCount
    
    Dim swDimAnn As SldWorks.Annotation
    Set swDimAnn = OrdDim.GetAnnotation
    swDimAnn.Select3 True, Nothing
    
    swDrawing.EditOrdinate
    
    If Qty > 1 Then
    
        If PrevDimCount + 1 = swView.GetDisplayDimensionCount Then
            
            Dim vDisplayDims As Variant
            vDisplayDims = swView.GetDisplayDimensions
            
            Dim swDisplayDim As SldWorks.DisplayDimension
            Set swDisplayDim = GetLastAddDisplayDimension(swView)

            If Not swDisplayDim Is Nothing Then
        
                Call AddQtyBracketsAndSuffixToDimension(swDisplayDim, Qty)
                
            End If
            
        End If
        
    End If

    swDrawing.SetPickMode
    swDrawing.ClearSelection2 True
    
End Sub

Sub SelectComponentOriginAndAddToOrdinateDimension(OrdDim As SldWorks.DisplayDimension, swComp As SldWorks.Component2, _
                Qty As Integer, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim PrevDimCount As Integer
    PrevDimCount = swView.GetDisplayDimensionCount
    
    Dim swDimAnn As SldWorks.Annotation
    Set swDimAnn = OrdDim.GetAnnotation
    swDimAnn.Select3 False, Nothing

    Call SelectComponentOrigin(swComp, swDrawing, swView, True)
    swDrawing.EditOrdinate
    
    If Qty > 1 Then
    
        If PrevDimCount + 1 = swView.GetDisplayDimensionCount Then
            
            Dim vDisplayDims As Variant
            vDisplayDims = swView.GetDisplayDimensions
            
            Dim swDisplayDim As SldWorks.DisplayDimension
            Set swDisplayDim = GetLastAddDisplayDimension(swView)

            If Not swDisplayDim Is Nothing Then
        
                Call AddQtyBracketsAndSuffixToDimension(swDisplayDim, Qty)
                
            End If
            
        End If
        
    End If

    swDrawing.SetPickMode
    swDrawing.ClearSelection2 True
    
End Sub

Function SelectComponentOrigin(swComp As SldWorks.Component2, swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, Append As Boolean) As Boolean
    
    Dim assyComponentName As String
    assyComponentName = swView.RootDrawingComponent.Component.Name2
    
    Dim assyDwgCompName As String
    assyDwgCompName = swView.RootDrawingComponent.Name

    
    Debug.Print "Point1@Origin@" & assyDwgCompName & "@" & swView.Name & "/" & swComp.Name2 & "@" & assyComponentName
    SelectComponentOrigin = swDrawing.Extension.SelectByID2("Point1@Origin@" & assyDwgCompName & "@" & swView.Name _
        & "/" & swComp.Name2 & "@" & assyComponentName, "EXTSKETCHPOINT", 0, 0, 0, Append, 0, Nothing, 0)

End Function


Function GetLastAddDisplayDimension(swView As SldWorks.View) As SldWorks.DisplayDimension

    Dim vDisplayDims As Variant
    vDisplayDims = swView.GetDisplayDimensions
    
    Dim DimNameToFind As String
    DimNameToFind = "D" & swView.GetDisplayDimensionCount + 4
    
    Dim i As Integer
    For i = UBound(vDisplayDims) To LBound(vDisplayDims) Step -1
        
        Dim swDisplayDim As SldWorks.DisplayDimension
        Set swDisplayDim = vDisplayDims(i)
        
        Dim swDimAnn As SldWorks.Annotation
        Set swDimAnn = swDisplayDim.GetAnnotation
        
        If swDimAnn.GetName = DimNameToFind Then
        
            Set GetLastAddDisplayDimension = swDisplayDim
            Exit For
            
        End If
    
    Next i

End Function
Function SelectAndAddDimension(swEnt1 As SldWorks.Entity, swEnt2 As SldWorks.Entity, swDrawing As SldWorks.ModelDoc2, _
            xPos As Double, yPos As Double, swView As SldWorks.View, Optional IsDual As Boolean = True) As SldWorks.DisplayDimension

    If Not (swEnt1 Is Nothing) And Not (swEnt2 Is Nothing) Then

        swDrawing.ClearSelection2 True
        
        swView.SelectEntity swEnt1, False
        swView.SelectEntity swEnt2, True

        Set SelectAndAddDimension = swDrawing.AddDimension2(xPos, yPos, 0)

        If Not SelectAndAddDimension Is Nothing Then

            SelectAndAddDimension.CenterText = True

            If IsDual Then

                SelectAndAddDimension.SetDual2 False, False

            End If

        End If

    End If

End Function

Sub AddCollinearRelation(swDrawing As SldWorks.DrawingDoc, swEdge As SldWorks.Edge, swSketchSegment As SldWorks.SketchSegment, swView As SldWorks.View)
    
    If Not (swEdge Is Nothing) And Not (swSketchSegment Is Nothing) Then
        
        swView.SelectEntity swEdge, False
        swSketchSegment.Select4 True, Nothing
                
        swDrawing.SketchAddConstraints "sgCOLINEAR"
        
    End If
    
End Sub
