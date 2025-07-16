Attribute VB_Name = "CrossMark"
Const LayerName As String = "BLOCKOUT SKETCH AND LEGENDS"
Public Const HatchLayName As String = "EPS HATCH"

Sub AddCrossMarkHatchAndItemNoCallOuts(BlockOutList As IArrListObject, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View)
    
    If (BlockOutList.Count > 0) Then
        
        Dim vBlockOuts As Variant
        vBlockOuts = BlockOutList.Items
        
        swView.FocusLocked = True

        Dim swLayerMgr As SldWorks.LayerMgr
        Set swLayerMgr = swDrawing.GetLayerManager
        
        Call CheckandAddLayer(LayerName, "BLOCKOUT SKETCH, NOTES & BALLOONS", swLayerMgr)
        Call CheckandAddLayer(HatchLayName, "FOAM HATCHES", swLayerMgr)
        
        Dim swHatchLayer As SldWorks.Layer
        Set swHatchLayer = swLayerMgr.GetLayer(HatchLayName)
        
        swHatchLayer.Color = RGB(192, 192, 192)
    
        Dim i As Integer
        For i = LBound(vBlockOuts) To UBound(vBlockOuts)
        
            Dim oBlockOut As IWeldBody
            Set oBlockOut = vBlockOuts(i)

            If oBlockOut.IsCircular Then
            

            
            ElseIf oBlockOut.IsRectangular Then
            
                Dim swSketchManager As SldWorks.SketchManager
                Set swSketchManager = swDrawing.SketchManager
                
                Dim vViewMinPoint As Variant
                vViewMinPoint = GetPointInViewSpace(oBlockOut.xMin, oBlockOut.yMin, 0, swView)
            
                Dim vViewMaxPoint As Variant
                vViewMaxPoint = GetPointInViewSpace(oBlockOut.xMax, oBlockOut.yMax, 0, swView)
                
                Call AddSketchSegmentsAndConstraints(swSketchManager, CDbl(vViewMinPoint(0)), CDbl(vViewMinPoint(1)), CDbl(vViewMaxPoint(0)), _
                    CDbl(vViewMaxPoint(1)), oBlockOut.LeftBottomVertex, oBlockOut.RightTopVertex, swDrawing, swView)
                
                Call AddSketchSegmentsAndConstraints(swSketchManager, CDbl(vViewMinPoint(0)), CDbl(vViewMaxPoint(1)), CDbl(vViewMaxPoint(0)), _
                    CDbl(vViewMinPoint(1)), oBlockOut.LeftTopVertex, oBlockOut.RightBottomVertex, swDrawing, swView)
                    

                
                Dim SelXPos As Double
                Dim SelYPos As Double
                Dim AnnXPos As Double
                Dim AnnYPos As Double
                
                Dim IsLeaderReq As Boolean
                IsLeaderReq = False
                
                Call GetBlockOutFoamAnnotationData(oBlockOut, SelXPos, SelYPos, AnnXPos, AnnYPos, IsLeaderReq)
                Call SelectAndAddItemNoAnnotation(oBlockOut.NormalFace, swDrawing, swView, SelXPos, SelYPos, AnnXPos, AnnYPos, IsLeaderReq)
                    
            Else
            
                Call AddHatchAndCallOutForFoams(oBlockOut, swDrawing, swView)
                    
            End If

        
        Next i
        
        Call UpdateHatchProperties(swView)
        swView.FocusLocked = False
   
    End If
    
End Sub

Sub GetBlockOutFoamAnnotationData(oBlockOut As IWeldBody, ByRef SelXPos As Double, ByRef SelYPos As Double, _
            ByRef AnnXPos As Double, ByRef AnnYPos As Double, ByRef IsLeaderReq As Boolean)

    Dim BlockOutLength As Double
    BlockOutLength = oBlockOut.xMax - oBlockOut.xMin
                
    Dim BlockOutWidth As Double
    BlockOutWidth = oBlockOut.yMax - oBlockOut.yMin
                
    If BlockOutLength < 0.0075 Or BlockOutWidth < 0.0075 Then
                
        IsLeaderReq = True
        If BlockOutLength > BlockOutWidth Then
        
            SelXPos = (oBlockOut.xMin + oBlockOut.xMax) / 2
            AnnXPos = SelXPos + 0.00125
            SelYPos = oBlockOut.yMax
            AnnYPos = SelYPos + 0.005
                        
        Else
                    
            SelXPos = oBlockOut.xMax
            AnnXPos = SelXPos + 0.005
            SelYPos = (oBlockOut.yMax + oBlockOut.yMin) / 2
            AnnYPos = SelYPos + 0.00125
                    
        End If
                
    Else
    
        If BlockOutLength > BlockOutWidth Then
                    
            SelXPos = oBlockOut.xMin
            SelYPos = (oBlockOut.yMax + oBlockOut.yMin) / 2
                        
        Else
                    
            SelXPos = (oBlockOut.xMin + oBlockOut.xMax) / 2
            SelYPos = (oBlockOut.yMax + oBlockOut.yMin) / 2
                    
        End If
        
        AnnXPos = SelXPos + 0.00125
        AnnYPos = SelYPos + 0.00125
              
    End If
    
End Sub


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






