Attribute VB_Name = "AddBalloons"

Sub AddBalloonForCoverPlates(vCoverPlates As Variant, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, oComp As IComp)
    
    
    If Not IsEmpty(vCoverPlates) Then
    
        Dim AllowableDiff As Double
        AllowableDiff = 0.0015875 * swView.ScaleDecimal
        
        Dim i As Integer
        For i = LBound(vCoverPlates) To UBound(vCoverPlates)
    
            Dim oCoverPlate As IWeldBody
            Set oCoverPlate = vCoverPlates(i)

            If Not i = 0 Then
            
                Dim PrevCoverPlate As IWeldBody
                Set PrevCoverPlate = vCoverPlates(i - 1)
                
                If Abs(PrevCoverPlate.xMin - oCoverPlate.xMin) <= AllowableDiff And _
                        Abs(PrevCoverPlate.yMin - oCoverPlate.yMin) <= AllowableDiff Then
                    
                    If PrevCoverPlate.zMin > oCoverPlate.zMin Then
                    
                        GoTo NextIter
                        
                    End If
                
                End If
                
            End If
            
            
            Dim SelXPos As Double
            Dim SelYPos As Double
            
            Dim AnnXPos As Double
            Dim AnnYPos As Double
            Dim swEdge As SldWorks.Edge
            
            Call GetBalloonPosData(oCoverPlate, swEdge, SelXPos, SelYPos, AnnXPos, AnnYPos, oComp, swView)
            
            Dim swAnnotation As SldWorks.Annotation
            Set swAnnotation = SelectAndAddAnnotation(swEdge, swDrawing, swView, SelXPos, _
               SelYPos, AnnXPos, AnnYPos, swBS_SplitCirc, "$PRPWLD:" & Chr(34) & "LEGEND" & Chr(34), "$PRPWLD:" & Chr(34) & "ITEM NO" & Chr(34))
            
            If Not swAnnotation Is Nothing Then
                
                swAnnotation.Layer = LayerName
                
            End If
NextIter:
    
        Next i
    
    End If

End Sub

Sub GetBalloonPosData(oCoverPlate As IWeldBody, ByRef swEdge As SldWorks.Edge, ByRef SelXPos As Double, SelYPos As Double, _
            ByRef AnnXPos As Double, ByRef AnnYPos As Double, oComp As IComp, swView As SldWorks.View)
    
    SelXPos = (oCoverPlate.xMin + oCoverPlate.xMax) / 2
    SelYPos = oCoverPlate.yMax
    AnnXPos = SelXPos
    
    If Abs(oComp.yMax - SelYPos) <= 0.015 Then
    
        AnnYPos = oComp.yMax + 0.0075
        
    Else
        
        AnnYPos = SelYPos + 0.01
        
    End If
        
    Set swEdge = GetEdgeInViewForBody(oCoverPlate.GetComponent, oCoverPlate, swView, True, True)
    
    Dim TopCoverPlate As IWeldBody
    Set TopCoverPlate = oCoverPlate.TopWeldBody
    
    Dim BottomCoverPlate As IWeldBody
    Set BottomCoverPlate = oCoverPlate.BottomWeldBody
    
    Dim LeftCoverPlate As IWeldBody
    Set LeftCoverPlate = oCoverPlate.LeftWeldBody
    
    Dim RightCoverPlate As IWeldBody
    Set RightCoverPlate = oCoverPlate.RightWeldBody
    
    Call GetAnnXPosWhenCalledOutVertically(oCoverPlate, LeftCoverPlate, RightCoverPlate, SelXPos, AnnXPos)
    
    Dim TopDiff As Double
    Dim BottomDiff As Double
     
    If Not TopCoverPlate Is Nothing And Not BottomCoverPlate Is Nothing Then

        BottomDiff = oCoverPlate.yMin - BottomCoverPlate.yMax
        TopDiff = TopCoverPlate.yMin - oCoverPlate.yMax
        
        If TopDiff > 0.0075 Or BottomDiff > 0.0075 Then
        
            If BottomDiff > TopDiff Then
            
                Call GetCallOutDataForBottomEdge(oCoverPlate, swEdge, SelYPos, AnnYPos, oComp, swView)
                
                
            ElseIf TopDiff > 0.0075 And TopDiff < 0.01 Then
            
                AnnYPos = SelYPos + 0.0075

            End If
            
        Else
            
            Call GetAnnPosDataForHorizontalCallout(oCoverPlate, swEdge, SelXPos, SelYPos, AnnXPos, AnnYPos, oComp, swView)
        
        End If

    ElseIf BottomCoverPlate Is Nothing And Not TopCoverPlate Is Nothing Then
        
        TopDiff = TopCoverPlate.yMin - oCoverPlate.yMax
        If TopDiff < 0.02 Or Abs(oComp.yMin - oCoverPlate.yMin) <= 0.01 Then

            Call GetCallOutDataForBottomEdge(oCoverPlate, swEdge, SelYPos, AnnYPos, oComp, swView)
            
        End If

    ElseIf TopCoverPlate Is Nothing And BottomCoverPlate Is Nothing Then

        If Abs(oComp.yMin - oCoverPlate.yMin) <= 0.01 Then

            Call GetCallOutDataForBottomEdge(oCoverPlate, swEdge, SelYPos, AnnYPos, oComp, swView)

        End If
   
    End If

End Sub

Sub GetAnnPosDataForHorizontalCallout(oCoverPlate As IWeldBody, ByRef swEdge As SldWorks.Edge, ByRef SelXPos As Double, SelYPos As Double, _
            ByRef AnnXPos As Double, ByRef AnnYPos As Double, oComp As IComp, swView As SldWorks.View)
            
    SelXPos = oCoverPlate.xMax
    SelYPos = (oCoverPlate.yMax + oCoverPlate.yMin) / 2
    AnnYPos = SelYPos
    
    If Abs(oComp.xMax - SelXPos) <= 0.015 Then
    
        AnnXPos = oComp.xMax + 0.005
        
    Else
        
        AnnXPos = SelXPos + 0.0075
        
    End If
        
    Set swEdge = oCoverPlate.GetRightEdge.GetEdge
    
    Dim TopCoverPlate As IWeldBody
    Set TopCoverPlate = oCoverPlate.TopCoverPlate
    
    Dim BottomCoverPlate As IWeldBody
    Set BottomCoverPlate = oCoverPlate.BottomCoverPlate
    
    Dim LeftCoverPlate As IWeldBody
    Set LeftCoverPlate = oCoverPlate.LeftCoverPlate
    
    Dim RightCoverPlate As IWeldBody
    Set RightCoverPlate = oCoverPlate.RightCoverPlate
    
    Call GetAnnYPosWhenCalledOutHorizontally(oCoverPlate, BottomCoverPlate, TopCoverPlate, SelYPos, AnnYPos)
    
    If Not RightCoverPlate Is Nothing And Not LeftCoverPlate Is Nothing Then
    
        Dim LeftDiff As Double
        LeftDiff = oCoverPlate.xMin - LeftCoverPlate.xMax
            
        Dim RightDiff As Double
        RightDiff = RightCoverPlate.xMin - oCoverPlate.xMax
        
        If RightDiff > 0.0075 Or LeftDiff > 0.0075 Then
        
            If LeftDiff > RightDiff Then
            
                Call GetCallOutDataForLeftEdge(oCoverPlate, swEdge, SelXPos, AnnXPos, oComp, swView)
                
                
            ElseIf RightDiff > 0.0075 And RightDiff < 0.01 Then
            
                AnnXPos = SelXPos + 0.0075

            End If
            
        End If

    ElseIf RightCoverPlate Is Nothing And LeftCoverPlate Is Nothing Then

        If Abs(oComp.xMax - oCoverPlate.xMax) > Abs(oCoverPlate.xMin - oComp.xMin) Then

            Call GetCallOutDataForLeftEdge(oCoverPlate, swEdge, SelXPos, AnnXPos, oComp, swView)
            
        End If

    ElseIf LeftCoverPlate Is Nothing Then

        Call GetCallOutDataForLeftEdge(oCoverPlate, swEdge, SelXPos, AnnXPos, oComp, swView)
   
    End If

End Sub

Sub GetCallOutDataForLeftEdge(oCoverPlate As IWeldBody, ByRef swEdge As SldWorks.Edge, _
                ByRef SelXPos As Double, ByRef AnnXPos As Double, oComp As IComp, swView As SldWorks.View)
                
    Set swEdge = GetEdgeInViewForBody(oCoverPlate.GetComponent, oCoverPlate, swView, False, False)
    SelXPos = oCoverPlate.xMin
    AnnXPos = SelXPos - 0.0075
    
    If Abs(oComp.yMin - SelYPos) <= 0.015 Then
    
        AnnXPos = oComp.xMin - 0.00375
        
    Else
        
        AnnXPos = SelXPos - 0.00625
        
    End If

End Sub

Sub GetCallOutDataForBottomEdge(oCoverPlate As IWeldBody, ByRef swEdge As SldWorks.Edge, _
                ByRef SelYPos As Double, ByRef AnnYPos As Double, oComp As IComp, swView As SldWorks.View)
                
    Set swEdge = GetEdgeInViewForBody(oCoverPlate.GetComponent, oCoverPlate, swView, True, False)
    SelYPos = oCoverPlate.yMin
    
    If Abs(oComp.yMin - SelYPos) <= 0.015 Then
    
        AnnYPos = oComp.yMin - 0.00375
        
    Else
        
        AnnYPos = SelYPos - 0.00625
        
    End If

End Sub

Sub GetAnnXPosWhenCalledOutVertically(oCoverPlate As IWeldBody, LeftCoverPlate As IWeldBody, _
        RightCoverPlate As IWeldBody, SelXPos As Double, ByRef AnnXPos As Double)

    If Abs(oCoverPlate.xMax - oCoverPlate.xMin) < 0.005 Then
            
        Dim LeftDiff As Double
        LeftDiff = GetMidPointDifference(oCoverPlate, LeftCoverPlate, "xMin", "xMax")
            
        Dim RightDiff As Double
        RightDiff = GetMidPointDifference(oCoverPlate, RightCoverPlate, "xMin", "xMax")
            
            
        If LeftDiff < 0.01 And RightDiff > 0.01 Then
            
            AnnXPos = SelXPos + 0.0025
                
                
        ElseIf LeftDiff > 0.01 And RightDiff < 0.01 Then
            
            AnnXPos = SelXPos - 0.00375
            
            
        End If

    End If
    
End Sub

Sub GetAnnYPosWhenCalledOutHorizontally(oCoverPlate As IWeldBody, BottomCoverPlate As IWeldBody, _
        TopCoverPlate As IWeldBody, SelYPos As Double, ByRef AnnYPos As Double)

    If oCoverPlate.GetLeftEdge.Length < 0.005 Then
            
        Dim BottomDiff As Double
        BottomDiff = GetMidPointDifference(oCoverPlate, BottomCoverPlate, "yMin", "yMax")
            
        Dim TopDiff As Double
        TopDiff = GetMidPointDifference(oCoverPlate, TopCoverPlate, "yMin", "yMax")
            
            
        If BottomDiff < 0.01 And TopDiff > 0.01 Then
            
            AnnYPos = SelYPos + 0.0025
                
                
        ElseIf BottomDiff > 0.01 And TopDiff < 0.01 Then
            
            AnnYPos = SelYPos - 0.0025
            
        End If

    End If
    
End Sub
Function GetMidPointDifference(oCoverPlate As IWeldBody, OtherCoverPlate As IWeldBody, _
                MinParam As String, MaxParam As String)

    If OtherCoverPlate Is Nothing Then
    
        GetMidPointDifference = 1
        
    Else
    
        GetMidPointDifference = Abs((CallByName(oCoverPlate, MinParam, VbGet) + CallByName(oCoverPlate, MaxParam, VbGet)) / 2 _
                - (CallByName(OtherCoverPlate, MinParam, VbGet) + CallByName(OtherCoverPlate, MaxParam, VbGet)) / 2)
        
    End If

End Function

Sub AddFloorPlateCallouts(xMinFloorDict As Scripting.Dictionary, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, OrgIsBottom As Boolean, oFloorComp As IComp)

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
        
        Dim IsBottom As Boolean
        IsBottom = OrgIsBottom
        
        PlateArrList.SortItems "yMin", False
        
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
       SelYPos As Double, AnnXPos As Double, AnnYPos As Double, Optional BalloonStyle As swBalloonStyle_e = 13, Optional CustomTextTop As String = "", _
        Optional CustomTextBottom As String = "") As SldWorks.Annotation

    Dim IsSelected As Boolean
    IsSelected = SelectEntityWithSelectData(swEnt, swView, swDrawing, SelXPos, SelYPos)
    
    If IsSelected Then

        Dim BalloonContent As swBalloonTextContent_e
    
        If CustomTextTop = "" And CustomTextBottom = "" Then
            BalloonContent = swBalloonTextPartNumberBOM
            
        Else
            
            BalloonContent = swBalloonTextCustom
            
        End If
        
        Dim swNote As SldWorks.Note
        Set swNote = swDrawing.InsertBOMBalloon2(BalloonStyle, swBF_Tightest, BalloonContent, _
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
        
        Debug.Print oFloorPlate.GetComponent.Name2
        
        Dim Gap As Double
        Gap = GetGapInfoForFloorPlateCallout(TopGap, BottomGap, IsBottom)
        
        Dim CallOutBlockOut As IBlockOut
        Dim IsFound As Boolean
        'Set CallOutBlockOut = GetBlockOutGreaterThanThisValInArrList(vBlockOutList, xPos, IsFound)
        
        'xPos = 0.75 * oFloorPlate.xMin + 0.25 * oFloorPlate.xMax
        
        'If IsFound Then
        
            Call xPosInCaseOfBlockOuts(vBlockOutList, oFloorPlate, xPos)

        'End If

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
    
        ArrList.SortItems "xMin", False
    
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


