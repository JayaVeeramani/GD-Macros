Attribute VB_Name = "CrossMark"

Function GetSketchContours(swSketch As SldWorks.Sketch, swComp As SldWorks.Component2, swView As SldWorks.View) As Variant

    Dim IsInit As Boolean
    IsInit = True
    
    Dim vContours() As IContourSketch
    
    If swSketch.GetSketchContourCount > 0 Then
        
        Dim vSketchContours As Variant
        vSketchContours = swSketch.GetSketchContours
        
        Dim vContourArrList As IArrListObject
        Set vContourArrList = New IArrListObject
        
        Dim i As Integer
        For i = LBound(vSketchContours) To UBound(vSketchContours)
                
            Dim swContour As SldWorks.SketchContour
            Set swContour = vSketchContours(i)
            
            If swContour.IsClosed And swContour.GetSketchSegmentsCount = 4 Then
                
                Dim vSketchSegs As Variant
                vSketchSegs = swContour.GetSketchSegments
                
                Dim swSketchSegment As SldWorks.SketchSegment
                Set swSketchSegment = vSketchSegs(0)
                
                If Not swSketchSegment.ConstructionGeometry Then
                
                    Dim swContourSketch As IContourSketch
                    Set swContourSketch = New IContourSketch
                    
                    
                    
                    swContourSketch.Initialize swContour, swSketch, swComp, swView
                
                    If swContourSketch.isRectangular Then
                    
                        vContourArrList.AddtoList swContourSketch
                        
                    End If
                    
                End If

            End If
        
        Next
        
    End If
    
    GetSketchContours = vContourArrList.Items

End Function

Sub AddCrossMarkAndDimensionsForContours(vContours As Variant, swDrawing As SldWorks.DrawingDoc, _
            swFeat As SldWorks.Feature, swSketch As SldWorks.Sketch, swView As SldWorks.View, _
                AssyName As String, oSubAssy As ISubAssy)
    
    swView.FocusLocked = True
    
    Dim OffsetX As Double
    Dim OffsetY As Double
    
    Dim ContourList As IArrListObject
    Set ContourList = New IArrListObject
    
    ContourList.AddItems vContours
    
    ContourList.SortItems "xMin", False

    Call GetOffsetValues(OffsetX, OffsetY, swDrawing, swView)
    
    Dim swEntity As SldWorks.Entity
    Set swEntity = swSketch
    
    Dim assyComponentName As String
    assyComponentName = swView.RootDrawingComponent.Component.Name2
    
    Dim assyDwgcompName As String
    assyDwgcompName = swView.RootDrawingComponent.Name
    
    Dim viewName As String
    viewName = swView.Name
    
    Dim sketchName As String
    sketchName = swFeat.Name
    
    Dim SelectionString2 As String
    SelectionString2 = sketchName & "@" & assyDwgcompName & "@" & viewName
    
    Dim SelectionString3 As String
    SelectionString3 = ExtractCompNameForSelectByID(assyComponentName, AssyName)
    
    Dim swSketchToModelTransform As SldWorks.MathTransform
    Set swSketchToModelTransform = swSketch.ModelToSketchTransform.Inverse
    
    Debug.Print AssyName
    
            
    Dim swSketchManager As SldWorks.SketchManager
    Set swSketchManager = swDrawing.SketchManager
    
    Dim ConsolidatedHorizontalDict As Scripting.Dictionary
    Set ConsolidatedHorizontalDict = New Scripting.Dictionary
    

    Dim i As Integer
    For i = LBound(vContours) To UBound(vContours)
    
        Dim swSketchContour As IContourSketch
        Set swSketchContour = vContours(i)

        Call AddSketchSegmentsAndConstraints(swDrawing, swSketchManager, OffsetX, OffsetY, swSketchContour.bottomLeftPoint, swSketchContour.topRightPoint, SelectionString2, SelectionString3)
        Call AddSketchSegmentsAndConstraints(swDrawing, swSketchManager, OffsetX, OffsetY, swSketchContour.TopLeftPoint, swSketchContour.BottomRightPoint, SelectionString2, SelectionString3)
        
        Dim xMin As Double
        xMin = Round(swSketchContour.xMin, 4)
        

        
        Call CheckAddToDict(ConsolidatedHorizontalDict, xMin, swSketchContour)
        
        
'        Dim swDisplayDim As SldWorks.DisplayDimension
'
'        If Abs(oSubAssy.StartComp.xMin - swSketchContour.xMin) < Abs(oSubAssy.EndComp.xMax - swSketchContour.xMax) Then
'
'            Call SelectLine(swDrawing, swSketchContour.LeftSketchLine, SelectionString2, SelectionString3, False)
'            swView.SelectEntity oSubAssy.StartEdge, True
'
'            Set swDisplayDim = swDrawing.AddHorizontalDimension2(oSubAssy.StartComp.xMin + 0.001, swSketchContour.yMin - 0.005, 0)
'            Call CenterAndManualParanthesis(swDisplayDim)
'
'
'            Call SelectLine(swDrawing, swSketchContour.RightSketchLine, SelectionString2, SelectionString3, False)
'            Set swDisplayDim = swDrawing.AddVerticalDimension2(swSketchContour.xMax + 0.005, swSketchContour.yMin + 0.001, 0)
'            Call CenterAndManualParanthesis(swDisplayDim)
'
'        Else
'
'            Call SelectLine(swDrawing, swSketchContour.RightSketchLine, SelectionString2, SelectionString3, False)
'            swView.SelectEntity oSubAssy.EndEdge, True
'
'            Set swDisplayDim = swDrawing.AddHorizontalDimension2(oSubAssy.EndComp.xMax - 0.001, swSketchContour.yMin - 0.005, 0)
'            Call CenterAndManualParanthesis(swDisplayDim)
'
'            Call SelectLine(swDrawing, swSketchContour.LeftSketchLine, SelectionString2, SelectionString3, False)
'            Set swDisplayDim = swDrawing.AddVerticalDimension2(swSketchContour.xMin - 0.005, swSketchContour.yMin + 0.001, 0)
'            Call CenterAndManualParanthesis(swDisplayDim)
'
'        End If
'
'        Call SelectLine(swDrawing, swSketchContour.BottomSketchLine, SelectionString2, SelectionString3, False)
'        swView.SelectEntity oSubAssy.BottomEdge, True
'        Set swDisplayDim = swDrawing.AddVerticalDimension2(swSketchContour.xMax + 0.005, oSubAssy.StartComp.yMin + 0.001, 0)
'        Call CenterAndManualParanthesis(swDisplayDim)
'
'        Call SelectLine(swDrawing, swSketchContour.TopSketchLine, SelectionString2, SelectionString3, False)
'        Set swDisplayDim = swDrawing.AddHorizontalDimension2(swSketchContour.xMin + 0.001, swSketchContour.yMax + 0.005, 0)
'        Call CenterAndManualParanthesis(swDisplayDim)
        
    Next i
    
    ContourList.SortItems "yMin", False
    
    Dim ConsolidatedVerticalDict As Scripting.Dictionary
    Set ConsolidatedVerticalDict = ConsolidateContoursVertically(ContourList.Items)
    
    Call AddHorizontalDimension(ConsolidatedHorizontalDict, oSubAssy, swDrawing, swView, SelectionString2, SelectionString3)
    Call AddVerticalDimension(ConsolidatedVerticalDict, oSubAssy, swDrawing, swView, SelectionString2, SelectionString3)
 
End Sub

Function ConsolidateContoursVertically(vContours As Variant) As Scripting.Dictionary

    Set ConsolidateContoursVertically = New Scripting.Dictionary
    
    Dim i As Integer
    For i = LBound(vContours) To UBound(vContours)
    
        Dim swSketchContour As IContourSketch
        Set swSketchContour = vContours(i)
    
        Dim yMin As Double
        yMin = Round(swSketchContour.yMin, 4)
        
        Call CheckAddToDict(ConsolidateContoursVertically, yMin, swSketchContour)
        
    Next i

End Function

Sub AddHorizontalDimension(Dict As Scripting.Dictionary, oSubAssy As ISubAssy, _
        swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, SelectionString2 As String, SelectionString3 As String)

    Dim vKeys As Variant
    vKeys = Dict.Keys
    
    Dim i As Integer
    For i = LBound(vKeys) To UBound(vKeys)
    
        Dim ArrList As IArrListObject
        Set ArrList = Dict.Item(vKeys(i))
        
        ArrList.SortItems "yMin", False
        
        Dim vContours As Variant
        vContours = ArrList.Items
        
        Dim PosSketchContour As IContourSketch
        Set PosSketchContour = vContours(0)
        
        Dim IsRight As Boolean
        Dim PosSketchLine As SldWorks.SketchSegment
        Dim AssyEdge As SldWorks.Edge
        Dim XDimPos As Double
        
        If Abs(oSubAssy.StartComp.xMin - PosSketchContour.xMin) < Abs(oSubAssy.EndComp.xMax - PosSketchContour.xMax) Then

            IsRight = False
            Set PosSketchLine = PosSketchContour.LeftSketchLine
            Set AssyEdge = oSubAssy.StartEdge
            XDimPos = oSubAssy.StartComp.xMin + 0.001
        
        Else

            IsRight = True
            Set PosSketchLine = PosSketchContour.RightSketchLine
            Set AssyEdge = oSubAssy.EndEdge
            XDimPos = oSubAssy.EndComp.xMax - 0.001
            
        End If
        
        
        Call SelectLine(swDrawing, PosSketchLine, SelectionString2, SelectionString3, False)
        swView.SelectEntity AssyEdge, True
        
        Dim swDisplayDim As SldWorks.DisplayDimension
        Set swDisplayDim = swDrawing.AddHorizontalDimension2(XDimPos, (PosSketchContour.yMax + PosSketchContour.yMin) / 2, 0)
        Call CenterAndManualParanthesis(swDisplayDim, ArrList.Count)
        
        Dim LengthContourDict As Scripting.Dictionary
        
        Dim LengthContourQtyDict As Scripting.Dictionary
        Set LengthContourQtyDict = New Scripting.Dictionary
        
        Set LengthContourDict = ConsolidateBasedOnWidthorLength(vContours, LengthContourQtyDict, "Length")

        Dim vLengthKeys As Variant
        vLengthKeys = LengthContourDict.Keys
        
        Dim j As Integer
        For j = LBound(vLengthKeys) To UBound(vLengthKeys)
        
            Dim swSketchContour As IContourSketch
            Set swSketchContour = LengthContourDict.Item(vLengthKeys(j))
            

            Call SelectLine(swDrawing, swSketchContour.BottomSketchLine, SelectionString2, SelectionString3, False)
            Set swDisplayDim = swDrawing.AddHorizontalDimension2(swSketchContour.xMax - 0.001, swSketchContour.yMin - 0.005, 0)
            Call CenterAndManualParanthesis(swDisplayDim, LengthContourQtyDict.Item(vLengthKeys(j)))

        Next j

    Next i
    
    
End Sub

Sub AddVerticalDimension(Dict As Scripting.Dictionary, oSubAssy As ISubAssy, _
        swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, SelectionString2 As String, SelectionString3 As String)

    Dim vKeys As Variant
    vKeys = Dict.Keys
    
    Dim i As Integer
    For i = LBound(vKeys) To UBound(vKeys)
    
        Dim ArrList As IArrListObject
        Set ArrList = Dict.Item(vKeys(i))
        
        ArrList.SortItems "xMin", False
        
        Dim vContours As Variant
        vContours = ArrList.Items
        
        Dim PosSketchContour As IContourSketch
        Set PosSketchContour = vContours(0)
        
        Dim XDimPos As Double
        Dim IsRight As Boolean
        
        If Abs(oSubAssy.StartComp.xMin - PosSketchContour.xMin) < Abs(oSubAssy.EndComp.xMax - PosSketchContour.xMax) Then
            
            XDimPos = PosSketchContour.xMax + 0.005
            IsRight = True
        
        Else
        
            XDimPos = PosSketchContour.xMin - 0.005
            IsRight = False
            
        End If
        
        
        Call SelectLine(swDrawing, PosSketchContour.BottomSketchLine, SelectionString2, SelectionString3, False)
        swView.SelectEntity oSubAssy.BottomEdge, True
        
        Dim swDisplayDim As SldWorks.DisplayDimension
        Set swDisplayDim = swDrawing.AddVerticalDimension2(XDimPos, oSubAssy.StartComp.yMin + 0.001, 0)
        Call CenterAndManualParanthesis(swDisplayDim, ArrList.Count)
        
        Dim widthContourDict As Scripting.Dictionary
        
        Dim widthContourQtyDict As Scripting.Dictionary
        Set widthContourQtyDict = New Scripting.Dictionary
        
        Set widthContourDict = ConsolidateBasedOnWidthorLength(vContours, widthContourQtyDict)

        Dim vWidthKeys As Variant
        vWidthKeys = widthContourDict.Keys
        
        Dim j As Integer
        For j = LBound(vWidthKeys) To UBound(vWidthKeys)
        
            Dim swSketchContour As IContourSketch
            Set swSketchContour = widthContourDict.Item(vWidthKeys(j))
            
            Dim swSketchLine As SldWorks.SketchSegment
            If IsRight Then

                Set swSketchLine = swSketchContour.RightSketchLine
                
            Else
                
                Set swSketchLine = swSketchContour.LeftSketchLine
                
            End If
            
            Call SelectLine(swDrawing, swSketchContour.RightSketchLine, SelectionString2, SelectionString3, False)
            Set swDisplayDim = swDrawing.AddVerticalDimension2(swSketchContour.xMax + 0.005, swSketchContour.yMin + 0.001, 0)
            Call CenterAndManualParanthesis(swDisplayDim, widthContourQtyDict.Item(vWidthKeys(j)))

        Next j

    Next i
    
    
End Sub

Function ConsolidateBasedOnWidthorLength(vContours As Variant, ByRef ConsolidatedQtyDict As Scripting.Dictionary, Optional SortParam As String = "Width") As Scripting.Dictionary
    
    Set ConsolidateBasedOnWidthorLength = New Scripting.Dictionary
    
    Dim j As Integer
    For j = LBound(vContours) To UBound(vContours)
        
        Dim swSketchContour As IContourSketch
        Set swSketchContour = vContours(j)
        
        Dim keyVal As Double
        keyVal = Round(CallByName(swSketchContour, SortParam, VbGet), 4)
        
        If ConsolidateBasedOnWidthorLength.Exists(keyVal) Then
        
            ConsolidatedQtyDict.Item(keyVal) = ConsolidatedQtyDict.Item(keyVal) + 1
            
        Else
        
            ConsolidateBasedOnWidthorLength.Add keyVal, swSketchContour
            ConsolidatedQtyDict.Add keyVal, 1
        
        End If

    Next j
 
End Function



Function CheckAddToDict(ByRef Dict As Scripting.Dictionary, keyVal As Double, swSketchContour As IContourSketch)
    
    
    Dim ArrList As IArrListObject
    
    If Dict.Exists(keyVal) Then

        Set ArrList = Dict.Item(keyVal)
        ArrList.AddtoList swSketchContour
        
    Else
    
        If Dict.Count > 0 Then
            
            If Abs(Dict.Keys(UBound(Dict.Keys)) - keyVal) <= 0.0001 Then
            
                Set ArrList = Dict.Item(Dict.Keys(UBound(Dict.Keys)))
                ArrList.AddtoList swSketchContour
                
            Else
                
                Set ArrList = New IArrListObject
                ArrList.AddtoList swSketchContour
                
                Dict.Add keyVal, ArrList
                
            End If
            
        Else
        
            Set ArrList = New IArrListObject
            ArrList.AddtoList swSketchContour
            
            Dict.Add keyVal, ArrList
            
        End If
        
    
    End If
    
End Function

Sub HorizontalDimensionsForContours(vContours As Variant, swDrawing As SldWorks.DrawingDoc, swSketchMgr As SldWorks.SketchManager)

    Dim i As Integer
    For i = LBound(vContours) To UBound(vContours)
        
        Dim swSketchContour As IContourSketch
        Set swSketchContour = vContours(i)
        
        If i = 0 Then
            
            
        Else
        
        
        End If
    
    Next i

End Sub

Sub GetOffsetValues(ByRef OffsetX As Double, ByRef OffsetZ As Double, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim values As Variant
    values = swView.Position

    Dim dOrigPt(2) As Double
    dOrigPt(0) = 0: dOrigPt(1) = 0: dOrigPt(2) = 0

    Dim vMathPt  As Variant
    vMathPt = GetTransformPoint(dOrigPt, swView.ModelToViewTransform)

    Dim swScale As Double
    swScale = swView.ScaleDecimal

    OffsetX = (values(0) - vMathPt(0)) / (swScale) ' * 0.0254)
    OffsetZ = (values(1) - vMathPt(1)) / (swScale) ' * 0.0254)

End Sub

Sub AddSketchSegmentsAndConstraints(swDrawing As SldWorks.DrawingDoc, swSketchManager As SldWorks.SketchManager, OffsetX As Double, OffsetY As Double, _
                FirstPoint As SldWorks.sketchPoint, SecondPoint As SldWorks.sketchPoint, SelectionString2 As String, SelectionString3 As String)

    Dim x1Val As Double
    Dim x2Val As Double
    Dim y1Val As Double
    Dim y2Val As Double
    
    Dim skSegment As SketchSegment
    Set skSegment = swSketchManager.CreateLine(FirstPoint.X - OffsetX, FirstPoint.Y - OffsetY, FirstPoint.Z, _
                        SecondPoint.X - OffsetX, SecondPoint.Y - OffsetY, SecondPoint.Z)
    skSegment.ConstructionGeometry = True
    skSegment.Layer = Layername
    
    Dim skLine As SldWorks.sketchLine
    Set skLine = skSegment

    Call AddConstraint(swDrawing, FirstPoint, skLine.GetStartPoint2, FirstPoint.X, FirstPoint.Y, FirstPoint.Z, SelectionString2, SelectionString3)
    Call AddConstraint(swDrawing, SecondPoint, skLine.GetEndPoint2, SecondPoint.X, SecondPoint.Y, SecondPoint.Z, SelectionString2, SelectionString3)
    
End Sub

Sub AddConstraint(swDrawing As SldWorks.DrawingDoc, sketchPoint As SldWorks.sketchPoint, linePoint As SldWorks.sketchPoint, xVal, yVal, _
    zVal, SelectionString2 As String, SelectionString3 As String)

    Dim bool As Boolean
    bool = swDrawing.Extension.SelectByID2("Point" & sketchPoint.GetID(1) & "@" & SelectionString2 _
        & SelectionString3, "EXTSKETCHPOINT", xVal, yVal, zVal, False, 0, Nothing, 0)

    If bool Then
        
        linePoint.Select4 True, Nothing
        swDrawing.SketchAddConstraints "sgCOINCIDENT"

    End If
    
 
End Sub

Function SelectLine(swDrawing As SldWorks.DrawingDoc, sketchLine As SldWorks.SketchSegment, _
        SelectionString2 As String, SelectionString3 As String, Append As Boolean) As Boolean

    SelectLine = swDrawing.Extension.SelectByID2("Line" & sketchLine.GetID(1) & "@" & SelectionString2 _
        & SelectionString3, "EXTSKETCHSEGMENT", 0, 0, 0, Append, 0, Nothing, 0)

End Function


Function ExtractCompNameForSelectByID(TopLevelCompName As String, ChildName As String)

    Dim vChildNames As Variant
    vChildNames = Split(ChildName, "/")
    
    Dim TempString As String
    
    Dim i As Integer
    For i = LBound(vChildNames) To UBound(vChildNames)
    
        If i = LBound(vChildNames) Then
    
            TempString = "/" & vChildNames(i) & "@" & TopLevelCompName
            
        Else
        
            TempString = TempString & "/" & vChildNames(i) & "@" & Left(vChildNames(i - 1), InStrRev(vChildNames(i - 1), "-") - 1)
            
        End If

    
    Next i
    
    Debug.Print TempString
    
    ExtractCompNameForSelectByID = TempString
    
End Function

Sub CenterAndManualParanthesis(swDisplayDim As SldWorks.DisplayDimension, Optional Qty As Long, Optional BottomText As String = "")
    
    If Not swDisplayDim Is Nothing Then
    
        swDisplayDim.CenterText = True
        
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, "("
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextSuffix, ")"
    
        If Not (BottomText = "") Then
        
            swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, BottomText
        
        End If
        
        If Qty > 1 Then
        
            swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, Qty & "X ("
            
        End If
    
    End If
    

End Sub
