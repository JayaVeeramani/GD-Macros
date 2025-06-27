Attribute VB_Name = "AddDimensions"

Dim AllowableDifference As Double

Sub SegregateAndAddDimensionVertically(xMinBlockOutDict As Scripting.Dictionary, xMaxPlateList As IArrListObject, _
            xMinFloorDict As Scripting.Dictionary, oFloorComp As IComp, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    Dim swFloorLeftEdge As SldWorks.Edge
    Set swFloorLeftEdge = GetEdgeInView(oFloorComp, swView, False, False)
    
    Dim EndFloorPlate As IComp
    Set EndFloorPlate = xMaxPlateList.Items(0)
    
    Dim FloorPlateRightEdge As SldWorks.Edge
    Set FloorPlateRightEdge = GetEdgeInView(EndFloorPlate, swView, False, True)
    
    Dim BottomSideEdges As IArrListObject
    Set BottomSideEdges = New IArrListObject
    
    Dim TopSideEdges As IArrListObject
    Set TopSideEdges = New IArrListObject
    
    Call SegregateVerticalEdges(xMinBlockOutDict, BottomSideEdges, TopSideEdges, oFloorComp)
    
    If BottomSideEdges.Count < TopSideEdges.Count Then
        
        Call AddFloorPlateEdgesToList(xMinFloorDict.Items, swView, FloorPlateRightEdge, BottomSideEdges, False, True)

    Else
    
        Call AddFloorPlateEdgesToList(xMinFloorDict.Items, swView, FloorPlateRightEdge, TopSideEdges, False, False)
    
    End If
    
    Dim BeforeDimCount As Integer
    BeforeDimCount = swView.GetDisplayDimensionCount
    
    Call ClearSelectionAndSelectEdges(swFloorLeftEdge, BottomSideEdges.Items, swDrawing, swView)
    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swHorizontalOrdinate, oFloorComp.xMax, oFloorComp.yMin - 0.01, 0
    
    Call ClearSelectionAndSelectEdges(swFloorLeftEdge, TopSideEdges.Items, swDrawing, swView)
    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swHorizontalOrdinate, oFloorComp.xMax, oFloorComp.yMax + 0.01, 0
    
    Call AddDimensionQtyToOrdinates(BeforeDimCount, xMinBlockOutDict, xMinFloorDict, _
                xMaxPlateList.Count, EndFloorPlate.xMax, swView, oFloorComp)
    
End Sub

Sub AddDimensionQtyToOrdinates(BeforeDimCount As Integer, BlockOutDict As Scripting.Dictionary, _
                FloorDict As Scripting.Dictionary, MaxPlateCount As Integer, MaxDimVal As Double, swView As SldWorks.View, _
                oFloorComp As IComp, Optional IsXDimension As Boolean = True)
                
    Dim vDisplayDims As Variant
    vDisplayDims = swView.GetDisplayDimensions
    
    AllowableDifference = swView.ScaleDecimal * 0.0015875
    
    If Not IsEmpty(vDisplayDims) Then
    
        Dim i As Integer
        For i = LBound(vDisplayDims) To UBound(vDisplayDims)
        
            Dim swDisplayDim As SldWorks.DisplayDimension
            Set swDisplayDim = vDisplayDims(i)

            If i > (BeforeDimCount - 1) Then
            
                Dim swDim As SldWorks.Dimension
                Set swDim = swDisplayDim.GetDimension2(0)
                
                Dim DimVal As Double
                DimVal = swDim.GetSystemValue2("")
                
                If IsXDimension Then
                
                    DimVal = oFloorComp.xMin + DimVal * swView.ScaleDecimal
                    
                Else
                    
                    DimVal = oFloorComp.yMin + DimVal * swView.ScaleDecimal
                    
                End If
                
                

            
                    Dim Qty As Integer
                    Dim IsQtyFound As Boolean
                    Qty = GetQtyForDimensions(BlockOutDict, FloorDict, DimVal, IsQtyFound)
                    
                    If IsQtyFound Then
                    
                        Call AddQtyBracketsAndSuffixToDimension(swDisplayDim, Qty, False)
                        
                    Else
                    
                        If Abs(DimVal - MaxDimVal) <= AllowableDifference Then
                            
                            Call AddQtyBracketsAndSuffixToDimension(swDisplayDim, MaxPlateCount, True, "FLOOR PLATE" & vbCrLf & "END")
                        
                        End If
                        
                    End If
                
                End If
                


        Next i
    
    End If

End Sub

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

Function GetQtyForDimensions(BlockOutDict As Scripting.Dictionary, FloorDict As Scripting.Dictionary, _
            CurrentDimVal As Double, ByRef IsQtyFound As Boolean) As Integer
        

    Dim Index As Integer
    Index = GetKeyIndexEqualToThisDimensionValue(BlockOutDict, CurrentDimVal, IsQtyFound)
    
    If IsQtyFound Then
    
        GetQtyForDimensions = BlockOutDict.Items(Index).Count
        
    Else
        
        Index = GetKeyIndexEqualToThisDimensionValue(FloorDict, CurrentDimVal, IsQtyFound)
        
        If IsQtyFound Then
        
            GetQtyForDimensions = FloorDict.Items(Index).Count
        
        End If
        
    End If

End Function

Function GetKeyIndexEqualToThisDimensionValue(Dict As Scripting.Dictionary, Val As Double, ByRef IsFound As Boolean) As Integer
    
    If Dict.Count > 0 Then
    
        Dim vKeys As Variant
        vKeys = Dict.Keys

        IsFound = False
        
        Dim i As Integer
        For i = LBound(vKeys) To UBound(vKeys)
            
            If Abs(CDbl(vKeys(i)) - Val) <= AllowableDifference Then
                
                GetKeyIndexEqualToThisDimensionValue = i
                IsFound = True
                Exit For
            
            End If
        
        Next i
        
    End If
    
End Function

Sub SegregateAndAddDimensionHorizontally(yMinBlockOutDict As Scripting.Dictionary, yMaxPlateList As IArrListObject, _
            yMinFloorDict As Scripting.Dictionary, oFloorComp As IComp, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    Dim swFloorBottomEdge As SldWorks.Edge
    Set swFloorBottomEdge = GetEdgeInView(oFloorComp, swView, True, False)

    Dim EndFloorPlate As IComp
    Set EndFloorPlate = yMaxPlateList.Items(0)

    Dim FloorPlateTopEdge As SldWorks.Edge
    Set FloorPlateTopEdge = GetEdgeInView(EndFloorPlate, swView, True, True)

    Dim LeftSideEdges As IArrListObject
    Set LeftSideEdges = New IArrListObject

    Dim RightSideEdges As IArrListObject
    Set RightSideEdges = New IArrListObject

    Call SegregateHorizontalEdges(yMinBlockOutDict, LeftSideEdges, RightSideEdges, oFloorComp)

    If LeftSideEdges.Count < RightSideEdges.Count Then

        Call AddFloorPlateEdgesToList(yMinFloorDict.Items, swView, FloorPlateTopEdge, LeftSideEdges, True, True)

    Else

        Call AddFloorPlateEdgesToList(yMinFloorDict.Items, swView, FloorPlateTopEdge, RightSideEdges, True, False)

    End If
    
    Dim BeforeDimCount As Integer
    BeforeDimCount = swView.GetDisplayDimensionCount
    
    Call ClearSelectionAndSelectEdges(swFloorBottomEdge, LeftSideEdges.Items, swDrawing, swView)
    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swVerticalOrdinate, oFloorComp.xMin - 0.01, oFloorComp.yMin, 0

    Call ClearSelectionAndSelectEdges(swFloorBottomEdge, RightSideEdges.Items, swDrawing, swView)
    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swVerticalOrdinate, oFloorComp.xMax + 0.01, oFloorComp.yMax, 0
    
    
    Call AddDimensionQtyToOrdinates(BeforeDimCount, yMinBlockOutDict, yMinFloorDict, _
                yMaxPlateList.Count, EndFloorPlate.yMax, swView, oFloorComp, False)
End Sub

Sub SegregateHorizontalEdges(yMinBlockOutDict As Scripting.Dictionary, LeftSideEdges As IArrListObject, _
         RightSideEdges As IArrListObject, oFloorComp As IComp)
    
    If yMinBlockOutDict.Count > 0 Then
        
        Dim vItems As Variant
        vItems = yMinBlockOutDict.Items
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim ArrList As IArrListObject
            Set ArrList = vItems(i)
            
            If ArrList.Count > 1 Then
            
                ArrList.SortItems "xMin", False
                
                Dim LowestBlockOut As IBlockOut
                Set LowestBlockOut = ArrList.Items(0)
                
                Dim HighestBlockOut As IBlockOut
                Set HighestBlockOut = ArrList.Items(UBound(ArrList.Items))
                
                Call SegregationBasedOnDistanceFromComponentEnd(oFloorComp, LowestBlockOut, HighestBlockOut, "xMin", "xMax", _
                        LeftSideEdges, RightSideEdges, "GetBottomEdge")
            
            Else
            
                Dim oBlockOut As IBlockOut
                Set oBlockOut = ArrList.Items(0)
            
                Dim BelowBlockOut As IBlockOut
                Set BelowBlockOut = oBlockOut.LeftBlockOut
                
                Dim AboveBlockOut As IBlockOut
                Set AboveBlockOut = oBlockOut.RightBlockOut
                
                If BelowBlockOut Is Nothing And AboveBlockOut Is Nothing Then
                
                    Call SegregationBasedOnDistanceFromComponentEnd(oFloorComp, oBlockOut, oBlockOut, "xMin", "xMax", _
                        LeftSideEdges, RightSideEdges, "GetBottomEdge")
                
                ElseIf BelowBlockOut Is Nothing And Not AboveBlockOut Is Nothing Then
                    
                    LeftSideEdges.AddtoList oBlockOut.GetBottomEdge.GetEdge
                    
                ElseIf AboveBlockOut Is Nothing And Not BelowBlockOut Is Nothing Then
                
                    RightSideEdges.AddtoList oBlockOut.GetBottomEdge.GetEdge
                    
                Else
                
                    If oBlockOut.yMin < BelowBlockOut.yMin And oBlockOut.yMin < AboveBlockOut.yMin Then
                        
                        Call SegregationBasedOnDistanceFromComponentEnd(oFloorComp, oBlockOut, oBlockOut, "xMin", "xMax", _
                            LeftSideEdges, RightSideEdges, "GetBottomEdge")
                            
                    ElseIf oBlockOut.yMin < BelowBlockOut.yMin Then
                    
                        LeftSideEdges.AddtoList oBlockOut.GetBottomEdge.GetEdge
                        
                    ElseIf oBlockOut.yMin < AboveBlockOut.yMin Then
                    
                        RightSideEdges.AddtoList oBlockOut.GetBottomEdge.GetEdge
                        
                    Else
                    
                        Call SegregationBasedOnDistanceFromComponentEnd(oFloorComp, oBlockOut, oBlockOut, "xMin", "xMax", _
                            LeftSideEdges, RightSideEdges, "GetBottomEdge")

                    End If

                End If
                
            End If
        
        Next i

    End If

End Sub

Sub SegregateVerticalEdges(xMinBlockOutDict As Scripting.Dictionary, BottomSideEdges As IArrListObject, _
         TopSideEdges As IArrListObject, oFloorComp As IComp)
    
    If xMinBlockOutDict.Count > 0 Then
        
        Dim vItems As Variant
        vItems = xMinBlockOutDict.Items
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim ArrList As IArrListObject
            Set ArrList = vItems(i)
            
            If ArrList.Count > 1 Then
            
                ArrList.SortItems "yMin", False
                
                Dim LowestBlockOut As IBlockOut
                Set LowestBlockOut = ArrList.Items(0)
                
                Dim HighestBlockOut As IBlockOut
                Set HighestBlockOut = ArrList.Items(UBound(ArrList.Items))
                
                Call SegregationBasedOnDistanceFromComponentEnd(oFloorComp, LowestBlockOut, HighestBlockOut, "yMin", "yMax", _
                        BottomSideEdges, TopSideEdges, "GetLeftEdge")
            
            Else
            
                Dim oBlockOut As IBlockOut
                Set oBlockOut = ArrList.Items(0)
            
                Dim BelowBlockOut As IBlockOut
                Set BelowBlockOut = oBlockOut.BottomBlockOut
                
                Dim AboveBlockOut As IBlockOut
                Set AboveBlockOut = oBlockOut.TopBlockOut
                
                If BelowBlockOut Is Nothing And AboveBlockOut Is Nothing Then
                
                    Call SegregationBasedOnDistanceFromComponentEnd(oFloorComp, oBlockOut, oBlockOut, "yMin", "yMax", _
                        BottomSideEdges, TopSideEdges, "GetLeftEdge")
                
                ElseIf BelowBlockOut Is Nothing And Not AboveBlockOut Is Nothing Then
                    
                    BottomSideEdges.AddtoList oBlockOut.GetLeftEdge.GetEdge
                    
                ElseIf AboveBlockOut Is Nothing And Not BelowBlockOut Is Nothing Then
                
                    TopSideEdges.AddtoList oBlockOut.GetLeftEdge.GetEdge
                    
                Else
                
                    If oBlockOut.xMin < BelowBlockOut.xMin And oBlockOut.xMin < AboveBlockOut.xMin Then
                        
                        Call SegregationBasedOnDistanceFromComponentEnd(oFloorComp, oBlockOut, oBlockOut, "yMin", "yMax", _
                            BottomSideEdges, TopSideEdges, "GetLeftEdge")
                            
                    ElseIf oBlockOut.xMin < BelowBlockOut.xMin Then
                    
                        BottomSideEdges.AddtoList oBlockOut.GetLeftEdge.GetEdge
                        
                    ElseIf oBlockOut.xMin < AboveBlockOut.xMin Then
                    
                        TopSideEdges.AddtoList oBlockOut.GetLeftEdge.GetEdge
                        
                    Else
                    
                        Call SegregationBasedOnDistanceFromComponentEnd(oFloorComp, oBlockOut, oBlockOut, "yMin", "yMax", _
                            BottomSideEdges, TopSideEdges, "GetLeftEdge")

                    End If

                End If
                
            End If
        
        Next i

    End If

End Sub

Sub SegregationBasedOnDistanceFromComponentEnd(oFloorComp As IComp, MinBlockOut As IBlockOut, MaxBlockOut As IBlockOut, _
        MinParam As String, MaxParam As String, LowEdgesList As IArrListObject, HighEdgesList As IArrListObject, EdgeName As String)

    If Abs(CallByName(oFloorComp, MinParam, VbGet) - CallByName(MinBlockOut, MinParam, VbGet)) _
                < Abs(CallByName(oFloorComp, MaxParam, VbGet) - CallByName(MaxBlockOut, MaxParam, VbGet)) Then
                    
        LowEdgesList.AddtoList CallByName(MinBlockOut, EdgeName, VbGet).GetEdge
                    
    Else
                
        HighEdgesList.AddtoList CallByName(MaxBlockOut, EdgeName, VbGet).GetEdge
                    
    End If
    
End Sub

Private Sub ClearSelectionAndSelectEdges(BeamLeftEdge As SldWorks.Edge, vEdges As Variant, _
            swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    If Not IsEmpty(vEdges) Then

        swDrawing.ClearSelection2 True
        swDrawing.SetPickMode
        swView.SelectEntity BeamLeftEdge, False

        Call SelectEdges(vEdges, swView)

    End If

End Sub

Function SelectEdges(vEdges As Variant, swView As SldWorks.View)

    Dim i As Integer
    For i = LBound(vEdges) To UBound(vEdges)
        
        Dim swEdge As SldWorks.Edge
        Set swEdge = vEdges(i)
            
        swView.SelectEntity swEdge, True
        
    Next i
    
End Function

Sub AddFloorPlateEdgesToList(vFloorCompsList As Variant, swView As SldWorks.View, swPlateEndEdge As SldWorks.Edge, _
        ArrList As IArrListObject, IsHorizontal As Boolean, IsBefore As Boolean)
    
    ArrList.AddtoList swPlateEndEdge
    
    Dim i As Integer
    For i = LBound(vFloorCompsList) To UBound(vFloorCompsList)
        
        Dim FloorCompList As IArrListObject
        Set FloorCompList = vFloorCompsList(i)
        
        If IsHorizontal Then
            
            FloorCompList.SortItems "xMin", False
        
        Else
        
            FloorCompList.SortItems "yMin", False
            
        End If
        
        Dim oComp As IComp
        
        If IsBefore Then

            Set oComp = FloorCompList.Items(0)
        Else
        
            Set oComp = FloorCompList.Items(UBound(FloorCompList.Items))
            
        End If
        
        Dim swEdge As SldWorks.Edge
        Set swEdge = GetEdgeInView(oComp, swView, IsHorizontal, False)
        
        ArrList.AddtoList swEdge
    
    Next i

End Sub
