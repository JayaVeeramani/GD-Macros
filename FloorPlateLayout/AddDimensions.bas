Attribute VB_Name = "AddDimensions"


Sub SegregateAndAddDimensionVertically(xMinBlockOutDict As Scripting.Dictionary, xMaxPlateList As IArrListObject, _
            FloorPlateList As IArrListObject, oFloorComp As IComp, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

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
        
        Call AddFloorPlateEdgesToList(FloorPlateList.Items, swView, FloorPlateRightEdge, BottomSideEdges, False)

    Else
    
        Call AddFloorPlateEdgesToList(FloorPlateList.Items, swView, FloorPlateRightEdge, TopSideEdges, False)
    
    End If
    
    Call ClearSelectionAndSelectEdges(swFloorLeftEdge, BottomSideEdges.Items, swDrawing, swView)
    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swHorizontalOrdinate, oFloorComp.xMax, DimYLoc, oFloorComp.yMax + 0.01
    
    Call ClearSelectionAndSelectEdges(swFloorLeftEdge, TopSideEdges.Items, swDrawing, swView)
    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swHorizontalOrdinate, oFloorComp.xMax, DimYLoc, oFloorComp.yMax + 0.01
    
End Sub

Sub SegregateAndAddDimensionHorizontally(yMinBlockOutDict As Scripting.Dictionary, yMaxPlateList As IArrListObject, _
            oFloorComp As IComp, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    Dim swFloorBottomEdge As SldWorks.Edge
    Set swFloorBottomEdge = GetEdgeInView(oFloorComp, swView, True, False)
    
    Dim LeftSideEdges As IArrListObject
    Set LeftSideEdges = New IArrListObject
    
    Dim RightSideEdges As IArrListObject
    Set RightSideEdges = New IArrListObject
    
    Call SegregateVerticalEdges(yMinBlockOutDict, LeftSideEdges, RightSideEdges, oFloorComp)
    
    Call AddVerticalBeamOrdinateDimensions(swFloorLeftEdge, BottomSideEdges.Items, swDrawing, swView, oFloorComp.xMax, oFloorComp.yMin - 0.01)
    Call AddVerticalBeamOrdinateDimensions(swFloorLeftEdge, TopSideEdges.Items, swDrawing, swView, oFloorComp.xMax, oFloorComp.yMax + 0.01)

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

Sub AddFloorPlateEdgesToList(vFloorComps As Variant, swView As SldWorks.View, swPlateEndEdge As SldWorks.Edge, _
        ArrList As IArrListObject, IsHorizontal As Boolean)
    
    ArrList.AddtoList swPlateEndEdge
    
    Dim i As Integer
    For i = LBound(vFloorComps) To UBound(vFloorComps)
        
        Dim oComp As IComp
        Set oComp = vFloorComps(i)
        
        Dim swEdge As SldWorks.Edge
        Set swEdge = GetEdgeInView(oComp, swView, IsHorizontal, False)
        
        ArrList.AddtoList swEdge
    
    Next i

End Sub
