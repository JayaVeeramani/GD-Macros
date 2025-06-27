Attribute VB_Name = "CrossMark"
Const LayerName As String = "BLOCKOUT SKETCH AND LEGENDS"
Const DimLayerName As String = "BLOCKOUT DIMENSIONS HIDDEN"

Sub AddCrossMarkAndBalloons(BlockOutList As IArrListObject, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, oComp As IComp)
    
    If (BlockOutList.Count > 0) Then
        
        BlockOutList.SortItems "Width"
        BlockOutList.SortItems "Length"
        
        Dim vBlockOuts As Variant
        vBlockOuts = BlockOutList.Items
        
        swView.FocusLocked = True
        
        Dim LegendAscii As Long
        LegendAscii = 65
        
        Dim IsAsciiMaxReached As Boolean
        IsAsciiMaxReached = False
        
        Dim LegendDict As Object
        Set LegendDict = CreateObject("Scripting.Dictionary")
    
        Dim ConsolidatedNote As String
        ConsolidatedNote = "<FONT style=B>LEGEND:<FONT style=RB>" & vbCrLf
        
        Dim swUserUnit As SldWorks.UserUnit
        Set swUserUnit = GetUserUnit
        
        Dim swLayerMgr As SldWorks.LayerMgr
        Set swLayerMgr = swDrawing.GetLayerManager
        
        Call CheckandAddLayer(LayerName, "BLOCKOUT SKETCH, NOTES & BALLOONS", swLayerMgr)
        Call CheckandAddLayer(DimLayerName, "HIDDEN BLOCKOUT DIMENSIONS", swLayerMgr)
    
        Dim i As Integer
        For i = LBound(vBlockOuts) To UBound(vBlockOuts)
        
            Dim oBlockOut As IBlockOut
            Set oBlockOut = vBlockOuts(i)
            
            Dim swSketchManager As SldWorks.SketchManager
            Set swSketchManager = swDrawing.SketchManager
            
            Dim vViewMinPoint As Variant
            vViewMinPoint = GetPointInViewSpace(oBlockOut.xMin, oBlockOut.yMin, 0, swView)
        
            Dim vViewMaxPoint As Variant
            vViewMaxPoint = GetPointInViewSpace(oBlockOut.xMax, oBlockOut.yMax, 0, swView)
            
            Dim LeftBottomVertex As SldWorks.Vertex
            Set LeftBottomVertex = oBlockOut.GetVertexPoint(oBlockOut.GetBottomEdge.GetEdge, oBlockOut.xMin)
            
            Dim RightTopVertex As SldWorks.Vertex
            Set RightTopVertex = oBlockOut.GetVertexPoint(oBlockOut.GetTopEdge.GetEdge, oBlockOut.xMax)
            
            Call AddSketchSegmentsAndConstraints(swSketchManager, CDbl(vViewMinPoint(0)), CDbl(vViewMinPoint(1)), CDbl(vViewMaxPoint(0)), _
                CDbl(vViewMaxPoint(1)), LeftBottomVertex, RightTopVertex, swDrawing, swView)
            
            Dim LeftTopVertex As SldWorks.Vertex
            Set LeftTopVertex = oBlockOut.GetVertexPoint(oBlockOut.GetTopEdge.GetEdge, oBlockOut.xMin)
            
            Dim RightBottomVertex As SldWorks.Vertex
            Set RightBottomVertex = oBlockOut.GetVertexPoint(oBlockOut.GetBottomEdge.GetEdge, oBlockOut.xMax)
            
            Call AddSketchSegmentsAndConstraints(swSketchManager, CDbl(vViewMinPoint(0)), CDbl(vViewMaxPoint(1)), CDbl(vViewMaxPoint(0)), _
                CDbl(vViewMinPoint(1)), LeftTopVertex, RightBottomVertex, swDrawing, swView)
            
            Dim BalloonIdExists As Boolean
            BalloonIdExists = IsBalloonIdAlreadyExists(LegendAscii, LegendDict, oBlockOut, swUserUnit, IsAsciiMaxReached)
    
            Call AddBalloon(oBlockOut, swView, swDrawing, oComp)
            
            If Not BalloonIdExists Then
    
                Call GetConsolidatedLegendNote(ConsolidatedNote, oBlockOut, swDrawing, swView)
    
            End If
        
        Next i
        
        swView.FocusLocked = False
        Call AddLegendNoteAtBottom(ConsolidatedNote, swDrawing)
    
        Dim swlayer As ILayer
        Set swlayer = swLayerMgr.GetLayer(DimLayerName)
    
        swlayer.Visible = False
        
    End If
    
End Sub

Function IsBalloonIdAlreadyExists(ByRef LegendAscii As Long, ByRef LegendDict As Object, oBlockOut As IBlockOut, _
    swUserUnit As SldWorks.UserUnit, ByRef IsAsciiMaxReached As Boolean) As Boolean

    Dim BlockoutSize As String
    BlockoutSize = swUserUnit.ConvertToUserUnit(oBlockOut.Length, True, True) & " X " & swUserUnit.ConvertToUserUnit(oBlockOut.Width, True, True)

    If LegendDict.Exists(BlockoutSize) Then

        oBlockOut.BalloonLegend = LegendDict.Item(BlockoutSize)
        IsBalloonIdAlreadyExists = True

    Else

        If False = IsAsciiMaxReached Then

            LegendDict.Add BlockoutSize, Chr(LegendAscii)
            oBlockOut.BalloonLegend = Chr(LegendAscii)

        Else

            LegendDict.Add BlockoutSize, Chr(LegendAscii) & Chr(LegendAscii)
            oBlockOut.BalloonLegend = Chr(LegendAscii) & Chr(LegendAscii)

        End If

        LegendAscii = GetValidAscii(LegendAscii, IsAsciiMaxReached)
        IsBalloonIdAlreadyExists = False

    End If

End Function

Function GetUserUnit() As SldWorks.UserUnit

    Dim swUserUnits As SldWorks.UserUnit
    Set swUserUnits = swApp.GetUserUnit(swUserUnitsType_e.swLengthUnit)

    swUserUnits.FractionBase = swFractionDisplay_e.swFRACTION
    swUserUnits.SpecificUnitType = swLengthUnit_e.swINCHES

    swUserUnits.RoundToFraction = True
    swUserUnits.FractionValue = 8
    
    Set GetUserUnit = swUserUnits
    
End Function

Function GetValidAscii(LegendAscii As Long, ByRef IsAsciiMaxReached As Boolean) As Long

    Dim IsNotValid As Boolean
    IsNotValid = True
    
    Do While IsNotValid
            
        LegendAscii = LegendAscii + 1
        If Not (LegendAscii = 73 Or LegendAscii = 79 Or LegendAscii = 81 Or LegendAscii = 83 Or LegendAscii = 88 Or LegendAscii = 90) Then
                
            IsNotValid = False
            
            If LegendAscii > 90 Then
            
                IsAsciiMaxReached = True
                LegendAscii = 65
                
            End If
            
        End If
            
    Loop

    GetValidAscii = LegendAscii
    
End Function

Sub AddSketchSegmentsAndConstraints(swSketchManager As SldWorks.SketchManager, xStart As Double, yStart As Double, _
        xEnd As Double, yEnd As Double, StartVertex As SldWorks.Vertex, EndVertex As SldWorks.Vertex, _
            swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim skSegment As SketchSegment
    Set skSegment = swSketchManager.CreateLine(xStart, yStart, 0, xEnd, yEnd, 0)
    skSegment.ConstructionGeometry = True
    skSegment.Layer = LayerName
    
    Dim skLine As SldWorks.SketchLine
    Set skLine = skSegment
    
    Call AddConstraint(skLine.GetStartPoint2, StartVertex, swDrawing, swView)
    Call AddConstraint(skLine.GetEndPoint2, EndVertex, swDrawing, swView)
    
End Sub

Function GetPointInViewSpace(xVal As Double, yVal As Double, zVal As Double, swView As SldWorks.View) As Variant
    
    Dim Point(2) As Double
    Point(0) = xVal
    Point(1) = yVal
    Point(2) = zVal
    
    Dim vPoint As Variant
    vPoint = Point
    
    GetPointInViewSpace = GetSheetPointInViewSpace(swView, Point)

End Function

Sub AddConstraint(sketchPoint As SldWorks.sketchPoint, swEnt As SldWorks.Entity, swDrawing As SldWorks.ModelDoc2, _
            swView As SldWorks.View)
    
    swDrawing.ClearSelection2 True
    
    swView.SelectEntity swEnt, False
    sketchPoint.Select4 True, Nothing
    swDrawing.SketchAddConstraints "sgCOINCIDENT"

End Sub


Sub AddBalloon(oBlockOut As IBlockOut, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, oComp As IComp)
    
    Dim SelXPos As Double
    Dim SelYPos As Double
    
    Dim AnnXPos As Double
    Dim AnnYPos As Double
    Dim swEdge As SldWorks.Edge
    
    Call GetBalloonPosData(oBlockOut, swEdge, SelXPos, SelYPos, AnnXPos, AnnYPos, oComp)
    
    Dim swAnnotation As SldWorks.Annotation
    Set swAnnotation = SelectAndAddAnnotation(swEdge, swDrawing, swView, SelXPos, _
       SelYPos, AnnXPos, AnnYPos, oBlockOut.BalloonLegend)
    
    If Not swAnnotation Is Nothing Then
        
        swAnnotation.Layer = LayerName
        oBlockOut.BalloonId = swDrawing.Extension.GetObjectId(swAnnotation)
        
    End If

End Sub

Sub GetBalloonPosData(oBlockOut As IBlockOut, ByRef swEdge As SldWorks.Edge, ByRef SelXPos As Double, SelYPos As Double, _
            ByRef AnnXPos As Double, ByRef AnnYPos As Double, oComp As IComp)
    
    SelXPos = (oBlockOut.xMin + oBlockOut.xMax) / 2
    SelYPos = oBlockOut.yMax
    AnnXPos = SelXPos
    
    If Abs(oComp.yMax - SelYPos) <= 0.015 Then
    
        AnnYPos = oComp.yMax + 0.005
        
    Else
        
        AnnYPos = SelYPos + 0.0075
        
    End If
        
    Set swEdge = oBlockOut.GetTopEdge.GetEdge
    
    Dim TopBlockOut As IBlockOut
    Set TopBlockOut = oBlockOut.TopBlockOut
    
    Dim BottomBlockOut As IBlockOut
    Set BottomBlockOut = oBlockOut.BottomBlockOut
    
    Dim LeftBlockOut As IBlockOut
    Set LeftBlockOut = oBlockOut.LeftBlockOut
    
    Dim RightBlockOut As IBlockOut
    Set RightBlockOut = oBlockOut.RightBlockOut
    
    Call GetAnnXPosWhenCalledOutVertically(oBlockOut, LeftBlockOut, RightBlockOut, SelXPos, AnnXPos)
    
    Dim TopDiff As Double
    Dim BottomDiff As Double
     
    If Not TopBlockOut Is Nothing And Not BottomBlockOut Is Nothing Then

        BottomDiff = oBlockOut.yMin - BottomBlockOut.yMax
        TopDiff = TopBlockOut.yMin - oBlockOut.yMax
        
        If TopDiff > 0.0075 Or BottomDiff > 0.0075 Then
        
            If BottomDiff > TopDiff Then
            
                Call GetCallOutDataForBottomEdge(oBlockOut, swEdge, SelYPos, AnnYPos, oComp)
                
                
            ElseIf TopDiff > 0.0075 And TopDiff < 0.01 Then
            
                AnnYPos = SelYPos + 0.0075

            End If
            
        Else
            
            Call GetAnnPosDataForHorizontalCallout(oBlockOut, swEdge, SelXPos, SelYPos, AnnXPos, AnnYPos, oComp)
        
        End If

    ElseIf BottomBlockOut Is Nothing And Not TopBlockOut Is Nothing Then
        
        TopDiff = TopBlockOut.yMin - oBlockOut.yMax
        If TopDiff < 0.02 Or Abs(oComp.yMin - oBlockOut.yMin) <= 0.01 Then

            Call GetCallOutDataForBottomEdge(oBlockOut, swEdge, SelYPos, AnnYPos, oComp)
            
        End If

    ElseIf TopBlockOut Is Nothing And BottomBlockOut Is Nothing Then

        If Abs(oComp.yMin - oBlockOut.yMin) <= 0.01 Then

            Call GetCallOutDataForBottomEdge(oBlockOut, swEdge, SelYPos, AnnYPos, oComp)

        End If
   
    End If

End Sub

Sub GetAnnPosDataForHorizontalCallout(oBlockOut As IBlockOut, ByRef swEdge As SldWorks.Edge, ByRef SelXPos As Double, SelYPos As Double, _
            ByRef AnnXPos As Double, ByRef AnnYPos As Double, oComp As IComp)
            
    SelXPos = oBlockOut.xMax
    SelYPos = (oBlockOut.yMax + oBlockOut.yMin) / 2
    AnnYPos = SelYPos
    
    If Abs(oComp.xMax - SelXPos) <= 0.015 Then
    
        AnnXPos = oComp.xMax + 0.005
        
    Else
        
        AnnXPos = SelXPos + 0.0075
        
    End If
        
    Set swEdge = oBlockOut.GetRightEdge.GetEdge
    
    Dim TopBlockOut As IBlockOut
    Set TopBlockOut = oBlockOut.TopBlockOut
    
    Dim BottomBlockOut As IBlockOut
    Set BottomBlockOut = oBlockOut.BottomBlockOut
    
    Dim LeftBlockOut As IBlockOut
    Set LeftBlockOut = oBlockOut.LeftBlockOut
    
    Dim RightBlockOut As IBlockOut
    Set RightBlockOut = oBlockOut.RightBlockOut
    
    Call GetAnnYPosWhenCalledOutHorizontally(oBlockOut, BottomBlockOut, TopBlockOut, SelYPos, AnnYPos)
    
    If Not RightBlockOut Is Nothing And Not LeftBlockOut Is Nothing Then
    
        Dim LeftDiff As Double
        LeftDiff = oBlockOut.xMin - LeftBlockOut.xMax
            
        Dim RightDiff As Double
        RightDiff = RightBlockOut.xMin - oBlockOut.xMax
        
        If RightDiff > 0.0075 Or LeftDiff > 0.0075 Then
        
            If LeftDiff > RightDiff Then
            
                Call GetCallOutDataForLeftEdge(oBlockOut, swEdge, SelXPos, AnnXPos, oComp)
                
                
            ElseIf RightDiff > 0.0075 And RightDiff < 0.01 Then
            
                AnnXPos = SelXPos + 0.0075

            End If
            
        End If

    ElseIf RightBlockOut Is Nothing And LeftBlockOut Is Nothing Then

        If Abs(oComp.xMax - oBlockOut.xMax) > Abs(oBlockOut.xMin - oComp.xMin) Then

            Call GetCallOutDataForLeftEdge(oBlockOut, swEdge, SelXPos, AnnXPos, oComp)
            
        End If

    ElseIf LeftBlockOut Is Nothing Then

        Call GetCallOutDataForLeftEdge(oBlockOut, swEdge, SelXPos, AnnXPos, oComp)
   
    End If

End Sub

Sub GetCallOutDataForLeftEdge(oBlockOut As IBlockOut, ByRef swEdge As SldWorks.Edge, _
                ByRef SelXPos As Double, ByRef AnnXPos As Double, oComp As IComp)
                
    Set swEdge = oBlockOut.GetLeftEdge.GetEdge
    SelXPos = oBlockOut.xMin
    AnnXPos = SelXPos - 0.0075
    
    If Abs(oComp.yMin - SelYPos) <= 0.015 Then
    
        AnnXPos = oComp.xMin - 0.00375
        
    Else
        
        AnnXPos = SelXPos - 0.00625
        
    End If

End Sub

Sub GetCallOutDataForBottomEdge(oBlockOut As IBlockOut, ByRef swEdge As SldWorks.Edge, _
                ByRef SelYPos As Double, ByRef AnnYPos As Double, oComp As IComp)
                
    Set swEdge = oBlockOut.GetBottomEdge.GetEdge
    SelYPos = oBlockOut.yMin
    
    If Abs(oComp.yMin - SelYPos) <= 0.015 Then
    
        AnnYPos = oComp.yMin - 0.00375
        
    Else
        
        AnnYPos = SelYPos - 0.00625
        
    End If

End Sub

Sub GetAnnXPosWhenCalledOutVertically(oBlockOut As IBlockOut, LeftBlockOut As IBlockOut, _
        RightBlockOut As IBlockOut, SelXPos As Double, ByRef AnnXPos As Double)

    If Abs(oBlockOut.xMax - oBlockOut.xMin) < 0.005 Then
            
        Dim LeftDiff As Double
        LeftDiff = GetMidPointDifference(oBlockOut, LeftBlockOut, "xMin", "xMax")
            
        Dim RightDiff As Double
        RightDiff = GetMidPointDifference(oBlockOut, RightBlockOut, "xMin", "xMax")
            
            
        If LeftDiff < 0.01 And RightDiff > 0.01 Then
            
            AnnXPos = SelXPos + 0.0025
                
                
        ElseIf LeftDiff > 0.01 And RightDiff < 0.01 Then
            
            AnnXPos = SelXPos - 0.00375
            
        End If

    End If
    
End Sub

Sub GetAnnYPosWhenCalledOutHorizontally(oBlockOut As IBlockOut, BottomBlockOut As IBlockOut, _
        TopBlockOut As IBlockOut, SelYPos As Double, ByRef AnnYPos As Double)

    If oBlockOut.GetLeftEdge.Length < 0.005 Then
            
        Dim BottomDiff As Double
        BottomDiff = GetMidPointDifference(oBlockOut, BottomBlockOut, "yMin", "yMax")
            
        Dim TopDiff As Double
        TopDiff = GetMidPointDifference(oBlockOut, TopBlockOut, "yMin", "yMax")
            
            
        If BottomDiff < 0.01 And TopDiff > 0.01 Then
            
            AnnYPos = SelYPos + 0.0025
                
                
        ElseIf BottomDiff > 0.01 And TopDiff < 0.01 Then
            
            AnnYPos = SelYPos - 0.0025
            
        End If

    End If
    
End Sub
Function GetMidPointDifference(oBlockOut As IBlockOut, OtherBlockOut As IBlockOut, _
                MinParam As String, MaxParam As String)

    If OtherBlockOut Is Nothing Then
    
        GetMidPointDifference = 1
        
    Else
    
        GetMidPointDifference = Abs((CallByName(oBlockOut, MinParam, VbGet) + CallByName(oBlockOut, MaxParam, VbGet)) / 2 _
                - (CallByName(OtherBlockOut, MinParam, VbGet) + CallByName(OtherBlockOut, MaxParam, VbGet)) / 2)
        
    End If

End Function

Sub GetConsolidatedLegendNote(ByRef ConsolidatedNote As String, oBlockOut As IBlockOut, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    Dim LengthDimName As String
    LengthDimName = AddDimensionAndGetDimensionName(oBlockOut.GetBottomEdge.GetEdge, swDrawing, _
                swView, (oBlockOut.xMin + oBlockOut.xMax) / 2, oBlockOut.yMin - 0.005)

    Dim WidthDimName As String
    WidthDimName = AddDimensionAndGetDimensionName(oBlockOut.GetLeftEdge.GetEdge, swDrawing, _
                swView, oBlockOut.xMin - 0.005, (oBlockOut.yMin + oBlockOut.yMax) / 2)

    ConsolidatedNote = ConsolidatedNote & "<OBJECT ID=" & oBlockOut.BalloonId & "> - " & _
           Chr(34) & LengthDimName & "@" & swView.Name & Chr(34) & " X " & _
           Chr(34) & WidthDimName & "@" & swView.Name & Chr(34) & vbCrLf


End Sub

Function AddDimensionAndGetDimensionName(swEnt As SldWorks.Entity, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, xPos As Double, yPos As Double)

    swView.SelectEntity swEnt, False
    
    Dim swDisplayDimension As SldWorks.DisplayDimension
    Set swDisplayDimension = swDrawing.AddDimension2(xPos, yPos, 0)
                
    Dim swDimAnnotation As SldWorks.Annotation
    Set swDimAnnotation = swDisplayDimension.GetAnnotation
    
    swDimAnnotation.Layer = DimLayerName
    
    'swDrawing.HideDimension
    AddDimensionAndGetDimensionName = swDisplayDimension.GetDimension2(0).Name
    
End Function

Sub AddLegendNoteAtBottom(NoteText As String, swDrawing As SldWorks.DrawingDoc)
        
    swDrawing.ClearSelection2 True
    
    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet
    
    swSheet.FocusLocked = True

    Dim LegNote As INote
    Set LegNote = swDrawing.InsertNote(NoteText)
    
    If Not LegNote Is Nothing Then
        Dim swAnnotation As IAnnotation
        Set swAnnotation = LegNote.GetAnnotation()

        swAnnotation.SetPosition 0.02201498, 0.11, 0
        swAnnotation.Layer = LayerName
        
    End If
    
    swSheet.FocusLocked = False

End Sub

Sub CheckandAddLayer(LayName As String, LayerDesc As String, swLayerMgr As SldWorks.LayerMgr)

    Dim vLayNames As Variant
    vLayNames = swLayerMgr.GetLayerList
    
    Dim IsLayerExists As Boolean
    
    Dim i As Integer
    For i = 0 To UBound(vLayNames)
    
        If vLayNames(i) = LayName Then
        
            IsLayerExists = True
            Exit For
            
        End If
        
    Next i
    
    If Not (IsLayerExists) Then
    
        swLayerMgr.AddLayer LayName, LayerDesc, 0, swLineStyles_e.swLineDEFAULT, swLineWeights_e.swLW_NONE
        
    End If
    
End Sub

 Function SelectEntityWithSelectData(swEnt As Object, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, _
                SelXPos As Double, SelYPos As Double) As Boolean

    Dim swSelectMgr As SldWorks.SelectionMgr
    Set swSelectMgr = swDrawing.SelectionManager
    
    Dim swSelectData As SldWorks.SelectData
    Set swSelectData = swSelectMgr.CreateSelectData


    swSelectData.View = swView
    swSelectData.X = SelXPos '(vStartPoint(0) + vEndPoint(0)) / 2
    swSelectData.Y = SelYPos 'vStartPoint(1)
    
    Dim swEntity As SldWorks.Entity
    Set swEntity = swEnt

    SelectEntityWithSelectData = swEntity.Select4(False, swSelectData)
    
End Function




