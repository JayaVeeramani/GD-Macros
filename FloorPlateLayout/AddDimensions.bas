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
    
    Call AddFloorPlateCallouts(xMinFloorDict, swDrawing, swView, BottomSideEdges.Count < TopSideEdges.Count, oFloorComp)
    
    If BottomSideEdges.Count < TopSideEdges.Count Then
        
        Call AddFloorPlateEdgesToList(xMinFloorDict.Items, swView, FloorPlateRightEdge, BottomSideEdges, False, True)

    Else
    
        Call AddFloorPlateEdgesToList(xMinFloorDict.Items, swView, FloorPlateRightEdge, TopSideEdges, False, False)
    
    End If
    
    Dim BeforeDimCount As Integer
    BeforeDimCount = swView.GetDisplayDimensionCount
    
    Call ClearSelectionAndSelectEdges(swFloorLeftEdge, BottomSideEdges.Items, swDrawing, swView)
    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swHorizontalOrdinate, oFloorComp.xMax, oFloorComp.yMin - 0.0085, 0
    
    Call ClearSelectionAndSelectEdges(swFloorLeftEdge, TopSideEdges.Items, swDrawing, swView)
    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swHorizontalOrdinate, oFloorComp.xMax, oFloorComp.yMax + 0.0085, 0
    
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
    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swVerticalOrdinate, oFloorComp.xMin - 0.0085, oFloorComp.yMin, 0

    Call ClearSelectionAndSelectEdges(swFloorBottomEdge, RightSideEdges.Items, swDrawing, swView)
    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swVerticalOrdinate, oFloorComp.xMax + 0.0085, oFloorComp.yMax, 0
    
    
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

Sub AddFloorPlateCallouts(xMinFloorDict As Scripting.Dictionary, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, IsBottom As Boolean, oFloorComp As IComp)

    Dim vItems As Variant
    vItems = xMinFloorDict.Items
    
    Dim i As Integer
    For i = LBound(vItems) To UBound(vItems)
    
        Dim PlateArrList As IArrListObject
        Set PlateArrList = vItems(i)
        
        Dim oFloorPlate As IComp
        Dim xPos As Double
        Dim yPos As Double
        Dim AnnXPos As Double
        Dim AnnYPos As Double
        
        Dim vFloorPlates As Variant
        vFloorPlates = PlateArrList.Items
                        
        Dim swAnnotation As SldWorks.Annotation
        
        If PlateArrList.Count = 1 Then
            
            Set oFloorPlate = vFloorPlates(0)
            
            If (oFloorPlate.yMin - oFloorComp.yMin) > 2 * 0.0254 * swView.ScaleDecimal And (oFloorComp.yMax - oFloorPlate.yMax) < 0.5 * 0.0254 * swView.ScaleDecimal Then
            
                IsBottom = False
                
            ElseIf (oFloorPlate.yMin - oFloorComp.yMin) < 0.5 * 0.0254 * swView.ScaleDecimal And (oFloorComp.yMax - oFloorPlate.yMax) > 2 * 0.0254 * swView.ScaleDecimal Then
            
                IsBottom = True
                
            End If
            
            Call GetFloorPlateCallOutPosition(oFloorPlate, xPos, yPos, AnnXPos, AnnYPos, IsBottom)
            Set swAnnotation = SelectAndAddAnnotation(oFloorPlate.VisibleFace, swDrawing, swView, xPos, _
                   yPos, AnnXPos, AnnYPos)
        
        Else

            Dim j As Integer
            For j = LBound(vFloorPlates) To UBound(vFloorPlates)

                Set oFloorPlate = vFloorPlates(j)
                Call GetFloorPlateCallOutPosition(oFloorPlate, xPos, yPos, AnnXPos, AnnYPos, j = 0)

                Set swAnnotation = SelectAndAddAnnotation(oFloorPlate.VisibleFace, swDrawing, swView, xPos, _
                   yPos, AnnXPos, AnnYPos)
                   
            Next j
            
        End If
        
    Next i

End Sub

Function SelectAndAddAnnotation(swEnt As Object, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, SelXPos As Double, _
       SelYPos As Double, AnnXPos As Double, AnnYPos As Double, Optional CustomTextTop As String = "", _
        Optional CustomTextBottom As String = "") As SldWorks.Annotation

    Dim IsSelected As Boolean
    IsSelected = SelectEntityWithSelectData(swEnt, swView, swDrawing, SelXPos, SelYPos)
    
    If IsSelected Then

        Dim BalloonContent As swBalloonTextContent_e
    
        If CustomTextTop = "" And CustomTextBottom = "" Then
            BalloonContent = swBalloonTextPartNumberBOM
        End If
        
        Dim swNote As SldWorks.Note
        Set swNote = swDrawing.InsertBOMBalloon2(swBS_Inspection, swBF_Tightest, BalloonContent, _
                                CustomTextTop, BalloonContent, CustomTextBottom)
            
        If Not swNote Is Nothing Then
        
            Dim swAnnotation As SldWorks.Annotation
            Set swAnnotation = swNote.GetAnnotation()
        
            swAnnotation.SetPosition AnnXPos, AnnYPos, 0
    
            Set SelectAndAddAnnotation = swAnnotation
        
        End If

    End If
End Function

Sub GetFloorPlateCallOutPosition(oFloorPlate As IComp, ByRef xPos As Double, ByRef yPos As Double, _
                ByRef AnnXPos As Double, ByRef AnnYPos As Double, IsBottom As Boolean)
    
   xPos = (oFloorPlate.xMin + oFloorPlate.xMax) / 2
   
    If oFloorPlate.BlockOutList.Count = 0 Then

        Call GetCallOutYPos(oFloorPlate, yPos, AnnYPos, 0.005, IsBottom)

    Else
    
        Dim vBlockOutList As IArrListObject
        Set vBlockOutList = oFloorPlate.BlockOutList
        
        Dim BottomGap As Double
        BottomGap = vBlockOutList.Items(0).yMin - oFloorPlate.yMin
        
        Dim TopGap As Double
        TopGap = oFloorPlate.yMax - vBlockOutList.Items(UBound(vBlockOutList.Items)).yMax
        
        Dim Gap As Double
        Gap = GetGapInfoForFloorPlateCallout(TopGap, BottomGap, IsBottom)
        
        Dim CallOutBlockOut As IBlockOut
        Dim IsFound As Boolean
        Set CallOutBlockOut = GetBlockOutGreaterThanThisValInArrList(vBlockOutList, xPos, IsFound)
        
        xPos = 0.75 * oFloorPlate.xMin + 0.25 * oFloorPlate.xMax
        
        If IsFound Then
        
            Call xPosInCaseOfBlockOuts(vBlockOutList, oFloorPlate, xPos)

        End If

        Call GetCallOutYPos(oFloorPlate, yPos, AnnYPos, Gap, IsBottom)

    End If
    AnnXPos = xPos
    
End Sub

Function GetGapInfoForFloorPlateCallout(TopGap As Double, BottomGap As Double, IsBottom As Boolean)
    
        
    Dim TempGap As Double
    TempGap = TopGap
    
    If IsBottom Then
        
        TempGap = BottomGap
            
    End If
    
    If TempGap < 0.0075 Then
        
        GetGapInfoForFloorPlateCallout = TempGap / 2
    
    Else
    
        GetGapInfoForFloorPlateCallout = 0.004
        
    End If
    
End Function

Function xPosInCaseOfBlockOuts(ArrList As IArrListObject, oFloorPlate As IComp, ByRef xPos As Double) As IBlockOut

    If ArrList.Count > 0 Then
    
        Dim vItems As Variant
        vItems = ArrList.Items

        Dim Gap As Double
        
        Dim TempGap As Double
        Dim TempPos As Double
        
        Gap = (vItems(0).xMin - oFloorPlate.xMin)
        xPos = (vItems(0).xMin + oFloorPlate.xMin) / 2

        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
            
            Dim oBlockOut As IBlockOut
            Set oBlockOut = vItems(i)
            
            If i = UBound(vItems) Then
            
                TempGap = oFloorPlate.xMax - oBlockOut.xMax
                TempPos = oFloorPlate.xMax - 0.4 * TempGap
                
            Else

                Dim NextBlockOut As IBlockOut
                Set NextBlockOut = vItems(i + 1)
                
                TempGap = NextBlockOut.xMin - oBlockOut.xMin
                TempPos = NextBlockOut.xMin - 0.4 * TempGap

            End If
            
            If TempGap > Gap Then
                
                Gap = TempGap
                xPos = TempPos
                
            End If
            
        Next i
        
    End If
    
End Function

Function GetBlockOutGreaterThanThisValInArrList(ArrList As IArrListObject, Val As Double, ByRef IsFound As Boolean) As IBlockOut
    
    ArrList.SortItems "xMin", False
    
    If ArrList.Count > 0 Then
    
        Dim vItems As Variant
        vItems = ArrList.Items

        IsFound = False
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
            
            Dim oBlockOut As IBlockOut
            Set oBlockOut = vItems(i)
            
            If oBlockOut.xMin > Val Then
            
                If Not i = 0 Then

                    Set GetBlockOutGreaterThanThisValInArrList = vItems(i - 1)
                    IsFound = True
                    
                End If
                
                Exit For
            
            End If
        
        Next i
        
    End If
    
End Function

Sub GetCallOutYPos(oFloorComp As IComp, ByRef yPos As Double, ByRef AnnYPos As Double, DistFromEnd As Double, IsBottom As Boolean)

    If IsBottom Then
        
        yPos = oFloorComp.yMin + DistFromEnd
        AnnYPos = oFloorComp.yMin - 0.025
        
    Else
    
        yPos = oFloorComp.yMax - DistFromEnd
        AnnYPos = oFloorComp.yMax + 0.025
        
    End If

End Sub
' Function InsertInspectionBalloon(swDrawing As SldWorks.DrawingDoc, AnnXPos As Double, AnnYPos As Double, Optional CustomTextTop As String = "", Optional CustomTextBottom As String = "") As SldWorks.Annotation
'
'    Dim BalloonContent As swBalloonTextContent_e
'
'    If CustomTextTop = "" And CustomTextBottom = "" Then
'        BalloonContent = swBalloonTextPartNumberBOM
'    End If
'
'    Dim swNote As SldWorks.Note
'    Set swNote = swDrawing.InsertBOMBalloon2(swBS_Inspection, swBF_Tightest, BalloonContent, _
'                            CustomTextTop, BalloonContent, CustomTextBottom)
'
'    If Not swNote Is Nothing Then
'
'        Dim swAnnotation As SldWorks.Annotation
'        Set swAnnotation = swNote.GetAnnotation()
'
'        swAnnotation.SetPosition AnnXPos, AnnYPos, 0
'
'        Set InsertInspectionBalloon = swAnnotation
'
'    End If
'
'End Function
